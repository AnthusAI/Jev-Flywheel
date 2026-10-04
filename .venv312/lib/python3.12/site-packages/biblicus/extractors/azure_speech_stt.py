"""
Azure Speech Services speech to text extractor plugin.

Uses Microsoft Azure Cognitive Services Speech API for audio transcription
with support for multiple languages and real-time streaming.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class AzureSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for Azure Speech Services speech to text extraction.

    :ivar language: Language code (en-US, es-ES, etc.).
    :vartype language: str
    :ivar region: Azure region (eastus, westus, etc.).
    :vartype region: str
    :ivar endpoint: Optional custom endpoint URL.
    :vartype endpoint: str or None
    :ivar profanity_option: Profanity filtering (masked, removed, raw).
    :vartype profanity_option: str
    :ivar enable_dictation: Enable dictation mode for punctuation.
    :vartype enable_dictation: bool
    """

    model_config = ConfigDict(extra="forbid")

    language: str = Field(default="en-US", min_length=1)
    region: str = Field(default="eastus", min_length=1)
    endpoint: Optional[str] = Field(default=None, min_length=1)
    profanity_option: str = Field(default="masked")
    enable_dictation: bool = Field(default=True)


class AzureSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio using Azure Speech Services.

    Uses Microsoft Azure Cognitive Services for high-quality speech recognition.
    Requires AZURE_SPEECH_KEY environment variable or configuration.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-azure-speech"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: AzureSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency or key is missing.
        """
        try:
            import azure.cognitiveservices.speech as speechsdk  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Azure Speech speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[azure]" or pip install azure-cognitiveservices-speech.'
            ) from import_error

        # Check for API key
        import os
        api_key = os.environ.get('AZURE_SPEECH_KEY')
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "Azure Speech speech to text extractor requires an Azure Speech API key. "
                "Set AZURE_SPEECH_KEY environment variable or configure it in ~/.biblicus/config.yml."
            )

        return AzureSpeechToTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item using Azure Speech Services.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: AzureSpeechToTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not audio.
        :rtype: ExtractedText or None
        :raises ExtractionSnapshotFatalError: If the optional dependency or key is missing.
        """
        _ = previous_extractions
        if not item.media_type.startswith("audio/"):
            return None

        parsed_config = (
            config
            if isinstance(config, AzureSpeechToTextExtractorConfig)
            else AzureSpeechToTextExtractorConfig.model_validate(config)
        )

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Azure Speech speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[azure]" or pip install azure-cognitiveservices-speech.'
            ) from import_error

        import os
        api_key = os.environ.get('AZURE_SPEECH_KEY')
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "Azure Speech speech to text extractor requires an Azure Speech API key. "
                "Set AZURE_SPEECH_KEY environment variable."
            )

        source_path = corpus.root / item.relpath

        # Configure speech recognizer
        if parsed_config.endpoint:
            speech_config = speechsdk.SpeechConfig(
                subscription=api_key,
                endpoint=parsed_config.endpoint
            )
        else:
            speech_config = speechsdk.SpeechConfig(
                subscription=api_key,
                region=parsed_config.region
            )

        speech_config.speech_recognition_language = parsed_config.language

        # Set profanity filtering
        if parsed_config.profanity_option == "masked":
            speech_config.set_profanity(speechsdk.ProfanityOption.Masked)
        elif parsed_config.profanity_option == "removed":
            speech_config.set_profanity(speechsdk.ProfanityOption.Removed)
        else:
            speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

        # Enable dictation mode for automatic punctuation
        if parsed_config.enable_dictation:
            speech_config.enable_dictation()

        # Create audio config from file
        audio_config = speechsdk.audio.AudioConfig(filename=str(source_path))

        # Create recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # Perform recognition
        result = speech_recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return ExtractedText(
                text=result.text.strip(),
                producer_extractor_id=self.extractor_id,
                metadata={
                    'language': parsed_config.language,
                    'region': parsed_config.region,
                    'confidence': getattr(result, 'confidence', None)
                }
            )
        elif result.reason == speechsdk.ResultReason.NoMatch:
            # No speech detected
            return ExtractedText(
                text="",
                producer_extractor_id=self.extractor_id,
                metadata={'reason': 'no_speech_detected'}
            )
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"Azure Speech recognition cancelled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            raise ExtractionSnapshotFatalError(error_msg)
        else:
            raise ExtractionSnapshotFatalError(
                f"Azure Speech recognition failed with reason: {result.reason}"
            )
