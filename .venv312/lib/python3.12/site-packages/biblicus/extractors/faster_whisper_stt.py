"""
Faster Whisper local speech to text extractor plugin.

Runs Whisper models locally using faster-whisper (CTranslate2 backend).
Supports latest Whisper large-v3 and other variants.
No API costs, fully offline capable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class FasterWhisperSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for faster-whisper local speech to text extraction.

    :ivar model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3).
    :vartype model_size: str
    :ivar device: Device to run on (cpu, cuda, auto).
    :vartype device: str
    :ivar compute_type: Computation type (int8, float16, float32).
    :vartype compute_type: str
    :ivar language: Optional language code for better accuracy.
    :vartype language: str or None
    :ivar beam_size: Beam size for decoding (higher = more accurate but slower).
    :vartype beam_size: int
    """

    model_config = ConfigDict(extra="forbid")

    model_size: str = Field(default="large-v3", min_length=1)
    device: str = Field(default="auto", min_length=1)
    compute_type: str = Field(default="int8", min_length=1)
    language: Optional[str] = Field(default=None, min_length=1)
    beam_size: int = Field(default=5, ge=1, le=10)


class FasterWhisperSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio using faster-whisper locally.

    Runs Whisper models on local hardware (CPU or GPU).
    No API costs, supports latest large-v3 model.
    Requires faster-whisper package.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-faster-whisper"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: FasterWhisperSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Faster Whisper speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[faster-whisper]" or pip install faster-whisper.'
            ) from import_error

        return FasterWhisperSpeechToTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item using faster-whisper.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: FasterWhisperSpeechToTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        _ = previous_extractions
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, FasterWhisperSpeechToTextExtractorConfig)
            else FasterWhisperSpeechToTextExtractorConfig.model_validate(config)
        )

        try:
            from faster_whisper import WhisperModel
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Faster Whisper speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[faster-whisper]" or pip install faster-whisper.'
            ) from import_error

        # Initialize model (cached per process)
        model = WhisperModel(
            parsed_config.model_size,
            device=parsed_config.device,
            compute_type=parsed_config.compute_type
        )

        source_path = corpus.root / item.relpath

        # Transcribe audio
        segments, info = model.transcribe(
            str(source_path),
            language=parsed_config.language,
            beam_size=parsed_config.beam_size
        )

        # Collect all segments
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text)

        transcript_text = " ".join(transcript_parts)

        return ExtractedText(
            text=transcript_text.strip(),
            producer_extractor_id=self.extractor_id,
            metadata={
                "model": parsed_config.model_size,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration
            }
        )
