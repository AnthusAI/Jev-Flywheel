"""
OpenAI-backed speech to text extractor plugin.

This extractor is implemented as an optional dependency so the core installation stays small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from ..user_config import resolve_openai_api_key
from .base import TextExtractor


class OpenAiSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for OpenAI speech to text extraction.

    :ivar model: OpenAI transcription model identifier.
    :vartype model: str
    :ivar response_format: OpenAI transcription response format.
    :vartype response_format: str
    :ivar language: Optional language code hint for transcription.
    :vartype language: str or None
    :ivar prompt: Optional prompt text to guide transcription.
    :vartype prompt: str or None
    :ivar no_speech_probability_threshold: Optional threshold for suppressing hallucinated transcripts.
    :vartype no_speech_probability_threshold: float or None
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="whisper-1", min_length=1)
    response_format: str = Field(default="json", min_length=1)
    language: Optional[str] = Field(default=None, min_length=1)
    prompt: Optional[str] = Field(default=None, min_length=1)
    no_speech_probability_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_no_speech_threshold(self) -> "OpenAiSpeechToTextExtractorConfig":
        if self.no_speech_probability_threshold is None:
            return self
        if self.response_format != "verbose_json":
            raise ValueError(
                "no_speech_probability_threshold requires response_format='verbose_json' "
                "so the transcription API returns per-segment no-speech probabilities"
            )
        return self


class OpenAiSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio items using the OpenAI API.

    This extractor is intended as a practical, hosted speech to text implementation.
    It skips non-audio items.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-openai"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: OpenAiSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency or required environment is missing.
        """
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "OpenAI speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[openai]".'
            ) from import_error

        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "OpenAI speech to text extractor requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )

        return OpenAiSpeechToTextExtractorConfig.model_validate(config)

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
        :type config: OpenAiSpeechToTextExtractorConfig
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
            if isinstance(config, OpenAiSpeechToTextExtractorConfig)
            else OpenAiSpeechToTextExtractorConfig.model_validate(config)
        )

        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "OpenAI speech to text extractor requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )

        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "OpenAI speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[openai]".'
            ) from import_error

        client = OpenAI(api_key=api_key)
        source_path = corpus.root / item.relpath
        with source_path.open("rb") as audio_handle:
            result = client.audio.transcriptions.create(
                file=audio_handle,
                model=parsed_config.model,
                response_format=parsed_config.response_format,
                language=parsed_config.language,
                prompt=parsed_config.prompt,
            )

        transcript_text: str
        no_speech_probability_threshold = parsed_config.no_speech_probability_threshold

        if isinstance(result, dict):
            transcript_text = str(result.get("text") or "")
            segments = result.get("segments")
            if (
                no_speech_probability_threshold is not None
                and isinstance(segments, list)
                and segments
            ):
                probabilities: list[float] = []
                for entry in segments:
                    if not isinstance(entry, dict):
                        continue
                    value = entry.get("no_speech_prob", entry.get("no_speech_probability"))
                    if isinstance(value, (int, float)):
                        probabilities.append(float(value))
                if probabilities and max(probabilities) >= no_speech_probability_threshold:
                    transcript_text = ""
        else:
            transcript_text = str(getattr(result, "text", "") or "")

        return ExtractedText(text=transcript_text.strip(), producer_extractor_id=self.extractor_id)
