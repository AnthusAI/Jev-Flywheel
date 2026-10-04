"""
Audio format converter transformer.

Converts audio files between formats (FLAC, WAV, MP3, etc.) for compatibility
with downstream extractors that have format restrictions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class AudioFormatConverterConfig(BaseModel):
    """
    Configuration for audio format conversion.

    :ivar target_format: Target audio format (wav, mp3, flac, ogg).
    :vartype target_format: str
    :ivar sample_rate: Optional target sample rate in Hz.
    :vartype sample_rate: int or None
    :ivar channels: Optional target channel count (1=mono, 2=stereo).
    :vartype channels: int or None
    """

    model_config = ConfigDict(extra="forbid")

    target_format: str = Field(default="wav", min_length=1)
    sample_rate: Optional[int] = Field(default=None, ge=8000, le=48000)
    channels: Optional[int] = Field(default=None, ge=1, le=2)


class AudioFormatConverterExtractor(TextExtractor):
    """
    Transformer that converts audio files to a target format.

    This is a pass-through extractor that converts the audio item to a different
    format and ingests it back into the corpus for downstream processing.
    It returns None to signal that no text was extracted at this stage.

    Requires pydub and ffmpeg to be installed.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "audio-format-converter"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate converter configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: AudioFormatConverterConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        try:
            from pydub import AudioSegment  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Audio format converter requires an optional dependency. "
                'Install it with pip install "biblicus[audio]" or pip install pydub.'
            ) from import_error

        parsed = AudioFormatConverterConfig.model_validate(config)

        # Validate target format
        valid_formats = {"wav", "mp3", "flac", "ogg", "m4a"}
        if parsed.target_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid target_format: {parsed.target_format}. "
                f"Must be one of: {', '.join(valid_formats)}"
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
        Convert audio item to target format.

        This extractor doesn't extract text - it transforms the audio data.
        It returns a special metadata payload that the pipeline can use to
        redirect subsequent stages to the converted audio.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: AudioFormatConverterConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: ExtractedText with conversion metadata, or None if not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency is missing.
        """
        _ = previous_extractions

        # Only process audio files
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, AudioFormatConverterConfig)
            else AudioFormatConverterConfig.model_validate(config)
        )

        try:
            from pydub import AudioSegment
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Audio format converter requires an optional dependency. "
                'Install it with pip install "biblicus[audio]" or pip install pydub.'
            ) from import_error

        source_path = corpus.root / item.relpath

        # Detect source format from media type
        source_format = self._detect_format(item.media_type)
        target_format = parsed_config.target_format.lower()

        inferred_from_unknown_type = (
            source_format == "wav" and "wav" not in item.media_type.lower()
        )

        # Skip conversion if already in target format and the media type was not an unknown default
        if source_format == target_format and not inferred_from_unknown_type:
            return ExtractedText(
                text="",  # No text extraction at this stage
                producer_extractor_id=self.extractor_id,
                metadata={
                    "conversion": "skipped",
                    "reason": "already_in_target_format",
                    "source_format": source_format,
                    "target_format": target_format
                }
            )

        # Load audio using pydub
        audio = AudioSegment.from_file(str(source_path), format=source_format)

        # Apply transformations if specified
        if parsed_config.sample_rate is not None:
            audio = audio.set_frame_rate(parsed_config.sample_rate)

        if parsed_config.channels is not None:
            audio = audio.set_channels(parsed_config.channels)

        # Export to target format in memory
        with tempfile.NamedTemporaryFile(
            suffix=f".{target_format}",
            delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            audio.export(str(temp_path), format=target_format)

        try:
            # Read converted audio data
            converted_data = temp_path.read_bytes()

            # Ingest converted audio back into corpus
            converted_item = corpus.ingest_item(
                data=converted_data,
                media_type=f"audio/{target_format}",
                source_uri=f"converted://{item.id}",
                tags=item.tags + ["converted", f"converted-from-{source_format}"]
            )

            # Return metadata about the conversion
            return ExtractedText(
                text="",  # No text extraction at this stage
                producer_extractor_id=self.extractor_id,
                metadata={
                    "conversion": "success",
                    "source_item_id": item.id,
                    "source_format": source_format,
                    "target_format": target_format,
                    "converted_item_id": converted_item.item_id,
                    "converted_relpath": converted_item.relpath,
                    "sample_rate": parsed_config.sample_rate,
                    "channels": parsed_config.channels,
                    "original_size_bytes": len(source_path.read_bytes()),
                    "converted_size_bytes": len(converted_data)
                }
            )
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)

    def _detect_format(self, media_type: str) -> str:
        """
        Detect audio format from media type.

        :param media_type: MIME media type.
        :type media_type: str
        :return: Audio format identifier.
        :rtype: str
        """
        media_type_lower = media_type.lower()

        if "flac" in media_type_lower:
            return "flac"
        elif "wav" in media_type_lower or "wave" in media_type_lower:
            return "wav"
        elif "mp3" in media_type_lower or "mpeg" in media_type_lower:
            return "mp3"
        elif "ogg" in media_type_lower or "vorbis" in media_type_lower:
            return "ogg"
        elif "m4a" in media_type_lower or "mp4" in media_type_lower:
            return "m4a"
        else:
            # Default to wav for unknown types
            return "wav"
