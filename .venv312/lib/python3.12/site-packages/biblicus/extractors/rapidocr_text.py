"""
RapidOCR-backed optical character recognition extractor plugin.

This extractor is an optional dependency. It exists as a practical default for extracting text
from image items without requiring a separate daemon.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class RapidOcrExtractorConfig(BaseModel):
    """
    Configuration for the RapidOCR extractor.

    :ivar min_confidence: Minimum per-line confidence to include in output.
    :vartype min_confidence: float
    :ivar joiner: Joiner used to combine recognized lines.
    :vartype joiner: str
    """

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    joiner: str = Field(default="\n")


class RapidOcrExtractor(TextExtractor):
    """
    Extractor plugin that performs optical character recognition on image items using RapidOCR.

    This extractor handles common image media types such as Portable Network Graphics and Joint Photographic Experts Group.
    It returns an empty extracted text artifact when the image is handled but no text is recognized.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "ocr-rapidocr"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: RapidOcrExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "RapidOCR extractor requires an optional dependency. "
                'Install it with pip install "biblicus[ocr]".'
            ) from import_error

        return RapidOcrExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text from an image item using optical character recognition.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: RapidOcrExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not an image.
        :rtype: ExtractedText or None
        """
        _ = previous_extractions
        media_type = item.media_type
        if not media_type.startswith("image/"):
            return None

        parsed_config = (
            config
            if isinstance(config, RapidOcrExtractorConfig)
            else RapidOcrExtractorConfig.model_validate(config)
        )

        from rapidocr_onnxruntime import RapidOCR

        source_path = corpus.root / item.relpath
        ocr = RapidOCR()
        result, _elapsed = ocr(str(source_path))

        if result is None:
            return ExtractedText(text="", producer_extractor_id=self.extractor_id)

        lines: list[str] = []
        confidences: list[float] = []
        for entry in result:
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            text_value = entry[1]
            confidence_value = entry[2]
            if not isinstance(text_value, str):
                continue
            if not isinstance(confidence_value, (int, float)):
                continue
            confidence = float(confidence_value)
            if confidence < parsed_config.min_confidence:
                continue
            cleaned = text_value.strip()
            if cleaned:
                lines.append(cleaned)
                confidences.append(confidence)

        text = parsed_config.joiner.join(lines).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        return ExtractedText(
            text=text,
            producer_extractor_id=self.extractor_id,
            confidence=avg_confidence,
        )
