"""
PaddleOCR-VL backed optical character recognition extractor plugin.

This extractor uses PaddleOCR-VL, a vision-language model that provides
improved optical character recognition accuracy especially for complex layouts and multilingual text.

The extractor supports both local inference and application programming interface based inference via
the inference backend abstraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..inference import ApiProvider, InferenceBackendConfig, InferenceBackendMode, resolve_api_key
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class PaddleOcrVlExtractorConfig(BaseModel):
    """
    Configuration for the PaddleOCR-VL extractor.

    :ivar retriever: Inference retriever configuration for local or application programming interface execution.
    :vartype retriever: InferenceBackendConfig
    :ivar min_confidence: Minimum confidence threshold for including text.
    :vartype min_confidence: float
    :ivar joiner: String used to join recognized text lines.
    :vartype joiner: str
    :ivar use_angle_cls: Whether to use angle classification for rotated text.
    :vartype use_angle_cls: bool
    :ivar lang: Language code for optical character recognition model.
    :vartype lang: str
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    retriever: InferenceBackendConfig = Field(
        default_factory=InferenceBackendConfig, alias="backend"
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    joiner: str = Field(default="\n")
    use_angle_cls: bool = Field(default=True)
    lang: str = Field(default="en")


class PaddleOcrVlExtractor(TextExtractor):
    """
    Extractor plugin using PaddleOCR-VL for optical character recognition.

    This extractor handles image media types and returns text with confidence scores.
    It supports both local inference and application programming interface based inference.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "ocr-paddleocr-vl"

    _model_cache: ClassVar[Dict[Tuple[str, bool], Any]] = {}

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: PaddleOcrVlExtractorConfig
        :raises ExtractionSnapshotFatalError: If required dependencies are missing.
        """
        import json

        parsed_config = {}
        for key, value in config.items():
            if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                try:
                    parsed_config[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_config[key] = value
            else:
                parsed_config[key] = value

        parsed = PaddleOcrVlExtractorConfig.model_validate(parsed_config)

        if parsed.retriever.mode == InferenceBackendMode.LOCAL:
            try:
                from paddleocr import PaddleOCR  # noqa: F401
            except ImportError as import_error:
                raise ExtractionSnapshotFatalError(
                    "PaddleOCR-VL extractor (local mode) requires paddleocr. "
                    'Install it with pip install "biblicus[paddleocr]".'
                ) from import_error
        else:
            # api_provider is guaranteed to be set by InferenceBackendConfig validator
            api_key = resolve_api_key(
                parsed.retriever.api_provider,
                config_override=parsed.retriever.api_key,
            )
            if api_key is None:
                provider_name = parsed.retriever.api_provider.value.upper()
                raise ExtractionSnapshotFatalError(
                    f"PaddleOCR-VL extractor (API mode) requires an API key for {provider_name}. "
                    f"Set {provider_name}_API_KEY environment variable or configure "
                    f"{parsed.retriever.api_provider.value} in user config."
                )

        return parsed

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text from an image using PaddleOCR-VL.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: PaddleOcrVlExtractorConfig
        :param previous_extractions: Prior stage outputs for this item.
        :type previous_extractions: list[ExtractionStageOutput]
        :return: Extracted text with confidence, or None for non-image items.
        :rtype: ExtractedText or None
        """
        _ = previous_extractions

        if not item.media_type.startswith("image/"):
            return None

        parsed_config = (
            config
            if isinstance(config, PaddleOcrVlExtractorConfig)
            else PaddleOcrVlExtractorConfig.model_validate(config)
        )

        source_path = corpus.root / item.relpath

        if parsed_config.retriever.mode == InferenceBackendMode.LOCAL:
            text, confidence = self._extract_local(source_path, parsed_config)
        else:
            api_key = resolve_api_key(
                parsed_config.retriever.api_provider,
                config_override=parsed_config.retriever.api_key,
            )
            text, confidence = self._extract_via_api(source_path, parsed_config, api_key)

        return ExtractedText(
            text=text,
            producer_extractor_id=self.extractor_id,
            confidence=confidence,
        )

    def _extract_local(
        self, source_path: Path, config: PaddleOcrVlExtractorConfig
    ) -> Tuple[str, Optional[float]]:
        """
        Perform local inference using PaddleOCR.

        :param source_path: Path to the image file.
        :type source_path: Path
        :param config: Parsed extractor configuration.
        :type config: PaddleOcrVlExtractorConfig
        :return: Tuple of extracted text and average confidence score.
        :rtype: tuple[str, float or None]
        """
        from paddleocr import PaddleOCR

        cache_key = (config.lang, config.use_angle_cls)
        ocr = PaddleOcrVlExtractor._model_cache.get(cache_key)
        if ocr is None:
            ocr = PaddleOCR(
                use_angle_cls=config.use_angle_cls,
                lang=config.lang,
            )
            PaddleOcrVlExtractor._model_cache[cache_key] = ocr
        result = ocr.ocr(str(source_path))

        if result is None or not result:
            return "", None

        lines: list[str] = []
        confidences: list[float] = []

        # Handle new PaddleOCR API (v4+) that returns dict with rec_texts/rec_scores
        for page_result in result:
            if page_result is None:
                continue

            # New API: dict with 'rec_texts' and 'rec_scores'
            if isinstance(page_result, dict):
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])

                for text_value, conf_value in zip(rec_texts, rec_scores):
                    if isinstance(conf_value, (int, float)):
                        confidence = float(conf_value)
                        if confidence >= config.min_confidence:
                            if isinstance(text_value, str) and text_value.strip():
                                lines.append(text_value.strip())
                                confidences.append(confidence)

            # Old API: list of [bbox, (text, confidence)]
            elif isinstance(page_result, (list, tuple)):
                for line_result in page_result:
                    if not isinstance(line_result, (list, tuple)) or len(line_result) < 2:
                        continue
                    text_info = line_result[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text_value = text_info[0]
                        conf_value = text_info[1]
                        if isinstance(conf_value, (int, float)):
                            confidence = float(conf_value)
                            if confidence >= config.min_confidence:
                                if isinstance(text_value, str) and text_value.strip():
                                    lines.append(text_value.strip())
                                    confidences.append(confidence)

        text = config.joiner.join(lines).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return text, avg_confidence

    def _extract_via_api(
        self, source_path: Path, config: PaddleOcrVlExtractorConfig, api_key: Optional[str]
    ) -> Tuple[str, Optional[float]]:
        """
        Perform inference via application programming interface.

        :param source_path: Path to the image file.
        :type source_path: Path
        :param config: Parsed extractor configuration.
        :type config: PaddleOcrVlExtractorConfig
        :param api_key: Application programming interface key for the provider.
        :type api_key: str or None
        :return: Tuple of extracted text and confidence score.
        :rtype: tuple[str, float or None]
        """
        if config.retriever.api_provider == ApiProvider.HUGGINGFACE:
            return self._extract_via_huggingface_api(source_path, config, api_key)
        else:
            return "", None

    def _extract_via_huggingface_api(
        self, source_path: Path, config: PaddleOcrVlExtractorConfig, api_key: Optional[str]
    ) -> Tuple[str, Optional[float]]:
        """
        Perform inference via HuggingFace Inference API.

        :param source_path: Path to the image file.
        :type source_path: Path
        :param config: Parsed extractor configuration.
        :type config: PaddleOcrVlExtractorConfig
        :param api_key: HuggingFace application programming interface key.
        :type api_key: str or None
        :return: Tuple of extracted text and confidence score.
        :rtype: tuple[str, float or None]
        """
        import base64

        import requests

        with open(source_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        headers = {"Authorization": f"Bearer {api_key}"}

        model_id = config.retriever.model_id or "PaddlePaddle/PaddleOCR-VL"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": image_data},
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        return self._parse_api_response(result, config)

    def _parse_api_response(
        self, result: Any, config: PaddleOcrVlExtractorConfig
    ) -> Tuple[str, Optional[float]]:
        """
        Parse application programming interface response.

        :param result: Application programming interface response data.
        :type result: Any
        :param config: Parsed extractor configuration.
        :type config: PaddleOcrVlExtractorConfig
        :return: Tuple of extracted text and confidence score.
        :rtype: tuple[str, float or None]
        """
        _ = config
        if isinstance(result, str):
            return result.strip(), None
        if isinstance(result, dict):
            text = result.get("generated_text", "")
            confidence = result.get("confidence")
            if isinstance(confidence, (int, float)):
                return text.strip(), float(confidence)
            return text.strip(), None
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                confidence = first.get("confidence")
                if isinstance(confidence, (int, float)):
                    return text.strip(), float(confidence)
                return text.strip(), None
        return "", None
