"""
Google Cloud Speech-to-Text extractor plugin.

Uses Google Cloud Speech-to-Text API for audio transcription with support
for multiple languages, automatic punctuation, and speaker diarization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class GoogleSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for Google Cloud Speech-to-Text extraction.

    :ivar language_code: Language code (en-US, es-ES, etc.).
    :vartype language_code: str
    :ivar model: Recognition model (default, latest_long, latest_short, etc.).
    :vartype model: str
    :ivar enable_automatic_punctuation: Add punctuation automatically.
    :vartype enable_automatic_punctuation: bool
    :ivar enable_speaker_diarization: Enable speaker identification.
    :vartype enable_speaker_diarization: bool
    :ivar diarization_speaker_count: Expected number of speakers.
    :vartype diarization_speaker_count: int or None
    :ivar enable_word_time_offsets: Include word-level timestamps.
    :vartype enable_word_time_offsets: bool
    :ivar profanity_filter: Filter profanity from results.
    :vartype profanity_filter: bool
    """

    model_config = ConfigDict(extra="forbid")

    language_code: str = Field(default="en-US", min_length=1)
    model: str = Field(default="default", min_length=1)
    enable_automatic_punctuation: bool = Field(default=True)
    enable_speaker_diarization: bool = Field(default=False)
    diarization_speaker_count: Optional[int] = Field(default=None, ge=2, le=6)
    enable_word_time_offsets: bool = Field(default=False)
    profanity_filter: bool = Field(default=False)


class GoogleSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio using Google Cloud Speech-to-Text.

    Uses Google Cloud Speech-to-Text API for high-quality speech recognition.
    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to
    a service account JSON file.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-google-speech"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: GoogleSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency or credentials are missing.
        """
        try:
            from google.cloud import speech  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Google Speech speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[google]" or pip install google-cloud-speech.'
            ) from import_error

        # Check for credentials
        import os
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path is None:
            raise ExtractionSnapshotFatalError(
                "Google Speech speech to text extractor requires Google Cloud credentials. "
                "Set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your service account JSON file."
            )

        return GoogleSpeechToTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item using Google Cloud Speech-to-Text.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: GoogleSpeechToTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency or credentials are missing.
        """
        _ = previous_extractions
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, GoogleSpeechToTextExtractorConfig)
            else GoogleSpeechToTextExtractorConfig.model_validate(config)
        )

        try:
            from google.cloud import speech
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Google Speech speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[google]" or pip install google-cloud-speech.'
            ) from import_error

        source_path = corpus.root / item.relpath

        # Initialize client
        client = speech.SpeechClient()

        # Read audio data
        with source_path.open('rb') as audio_file:
            audio_data = audio_file.read()

        # Detect audio encoding from media type
        encoding = self._detect_encoding(item.media_type)

        # Configure audio
        audio = speech.RecognitionAudio(content=audio_data)

        # Configure recognition
        recognition_config = speech.RecognitionConfig(
            encoding=encoding,
            language_code=parsed_config.language_code,
            model=parsed_config.model,
            enable_automatic_punctuation=parsed_config.enable_automatic_punctuation,
            profanity_filter=parsed_config.profanity_filter,
        )

        # Add word time offsets if requested
        if parsed_config.enable_word_time_offsets:
            recognition_config.enable_word_time_offsets = True

        # Add speaker diarization if requested
        if parsed_config.enable_speaker_diarization:
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
            )
            if parsed_config.diarization_speaker_count is not None:
                diarization_config.min_speaker_count = parsed_config.diarization_speaker_count
                diarization_config.max_speaker_count = parsed_config.diarization_speaker_count

            recognition_config.diarization_config = diarization_config

        # Perform recognition
        response = client.recognize(config=recognition_config, audio=audio)

        # Extract transcript
        transcript_parts = []
        metadata = {
            'language_code': parsed_config.language_code,
            'model': parsed_config.model
        }

        for result in response.results:
            if result.alternatives:
                alternative = result.alternatives[0]
                transcript_parts.append(alternative.transcript)

                # Add confidence if available
                if hasattr(alternative, 'confidence'):
                    if 'confidences' not in metadata:
                        metadata['confidences'] = []
                    metadata['confidences'].append(alternative.confidence)

        transcript_text = " ".join(transcript_parts)

        return ExtractedText(
            text=transcript_text.strip(),
            producer_extractor_id=self.extractor_id,
            metadata=metadata
        )

    def _detect_encoding(self, media_type: str) -> int:
        """
        Detect Google Cloud Speech encoding from MIME type.

        :param media_type: MIME media type.
        :type media_type: str
        :return: Google Cloud Speech encoding constant.
        :rtype: int
        """
        from google.cloud import speech

        media_type_lower = media_type.lower()

        if 'flac' in media_type_lower:
            return speech.RecognitionConfig.AudioEncoding.FLAC
        elif 'wav' in media_type_lower or 'wave' in media_type_lower:
            return speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif 'mp3' in media_type_lower or 'mpeg' in media_type_lower:
            return speech.RecognitionConfig.AudioEncoding.MP3
        elif 'ogg' in media_type_lower or 'opus' in media_type_lower:
            return speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        elif 'webm' in media_type_lower:
            return speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        else:
            return speech.RecognitionConfig.AudioEncoding.LINEAR16  # default
