"""
Aldea-backed speech to text extractor plugin.

This extractor is implemented as an optional dependency so the core installation stays small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from ..user_config import resolve_aldea_api_key
from .base import TextExtractor

ALDEA_LISTEN_URL = "https://api.aldea.ai/v1/listen"


class AldeaSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for Aldea speech to text extraction.

    :ivar language: Optional language code hint for transcription (BCP-47).
    :vartype language: str or None
    :ivar diarization: Whether to enable speaker diarization.
    :vartype diarization: bool
    :ivar timestamps: Whether to include per-word timestamps in response.
    :vartype timestamps: bool
    :ivar api_key: Optional override for the Aldea API key.
    :vartype api_key: str or None
    :ivar url: Endpoint for the Aldea transcription API.
    :vartype url: str
    """

    model_config = ConfigDict(extra="forbid")

    language: Optional[str] = Field(default=None, min_length=1)
    diarization: bool = Field(default=False)
    timestamps: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None, min_length=1)
    url: str = Field(default=ALDEA_LISTEN_URL, min_length=1)


class AldeaSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio items using the Aldea API.

    This extractor is intended as a practical, hosted speech to text implementation.
    It skips non-audio items.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-aldea"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: AldeaSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency or required environment is missing.
        """
        parsed = AldeaSpeechToTextExtractorConfig.model_validate(config)
        api_key = parsed.api_key or resolve_aldea_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "Aldea speech to text extractor requires an Aldea API key. "
                "Set ALDEA_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "aldea.api_key."
            )

        return parsed.model_copy(update={"api_key": api_key})

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: AldeaSpeechToTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency or required configuration is missing.
        """
        _ = previous_extractions
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, AldeaSpeechToTextExtractorConfig)
            else AldeaSpeechToTextExtractorConfig.model_validate(config)
        )

        api_key = parsed_config.api_key or resolve_aldea_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "Aldea speech to text extractor requires an Aldea API key. "
                "Set ALDEA_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "aldea.api_key."
            )

        try:
            import httpx  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Aldea speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[aldea]".'
            ) from import_error

        source_path = corpus.root / item.relpath
        try:
            with source_path.open("rb") as audio_handle:
                audio_data = audio_handle.read()
        except FileNotFoundError:
            return ExtractedText(text="", producer_extractor_id=self.extractor_id, metadata={"aldea": {}})

        params: Dict[str, Any] = {}
        if parsed_config.language is not None:
            params["language"] = parsed_config.language
        if parsed_config.diarization:
            params["diarization"] = "true"

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
        }
        if parsed_config.timestamps:
            headers["timestamps"] = "true"

        import httpx

        response = httpx.post(
            parsed_config.url,
            content=audio_data,
            params=params if params else None,
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        response_payload = response.json()

        transcript_text = ""
        results = response_payload.get("results")
        if isinstance(results, dict):
            channels = results.get("channels")
            if isinstance(channels, list) and len(channels) > 0:
                alternatives = channels[0].get("alternatives") if isinstance(channels[0], dict) else None
                if isinstance(alternatives, list) and len(alternatives) > 0:
                    first_alt = alternatives[0]
                    if isinstance(first_alt, dict):
                        transcript_text = str(first_alt.get("transcript") or "")

        return ExtractedText(
            text=transcript_text.strip(),
            producer_extractor_id=self.extractor_id,
            metadata={"aldea": response_payload},
        )
