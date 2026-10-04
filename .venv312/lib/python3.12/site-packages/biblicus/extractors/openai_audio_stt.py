"""
OpenAI GPT-4o Audio speech to text extractor plugin.

Uses GPT-4o's native audio understanding capabilities for transcription.
Different from Whisper - uses multimodal model that understands audio context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from ..user_config import resolve_openai_api_key
from .base import TextExtractor


class OpenAiAudioSpeechToTextExtractorConfig(BaseModel):
    """
    Configuration for OpenAI GPT-4o Audio speech to text extraction.

    :ivar model: OpenAI model identifier (gpt-4o-audio-preview, etc.).
    :vartype model: str
    :ivar prompt: Optional prompt to guide transcription style.
    :vartype prompt: str or None
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="gpt-4o-audio-preview", min_length=1)
    prompt: Optional[str] = Field(
        default="Transcribe this audio accurately, preserving all spoken words.",
        min_length=1
    )


class OpenAiAudioSpeechToTextExtractor(TextExtractor):
    """
    Extractor plugin that transcribes audio using GPT-4o Audio API.

    Uses GPT-4o's multimodal capabilities for audio understanding.
    May provide better context awareness than pure ASR models.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "stt-openai-audio"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure prerequisites are available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: OpenAiAudioSpeechToTextExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency or required environment is missing.
        """
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "OpenAI audio speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[openai]".'
            ) from import_error

        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "OpenAI audio speech to text extractor requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )

        return OpenAiAudioSpeechToTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Transcribe an audio item using GPT-4o Audio.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: OpenAiAudioSpeechToTextExtractorConfig
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
            if isinstance(config, OpenAiAudioSpeechToTextExtractorConfig)
            else OpenAiAudioSpeechToTextExtractorConfig.model_validate(config)
        )

        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ExtractionSnapshotFatalError(
                "OpenAI audio speech to text extractor requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )

        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "OpenAI audio speech to text extractor requires an optional dependency. "
                'Install it with pip install "biblicus[openai]".'
            ) from import_error

        client = OpenAI(api_key=api_key)
        source_path = corpus.root / item.relpath

        # Determine source format from media type
        media_type = item.media_type.lower()
        source_format = None
        if 'flac' in media_type:
            source_format = 'flac'
        elif 'wav' in media_type:
            source_format = 'wav'
        elif 'mp3' in media_type:
            source_format = 'mp3'
        elif 'ogg' in media_type:
            source_format = 'ogg'
        else:
            source_format = 'wav'

        # GPT-4o Audio only supports wav and mp3
        # Convert FLAC/OGG to WAV in-memory if needed
        if source_format in ('flac', 'ogg'):
            try:
                import io

                from pydub import AudioSegment
            except ImportError:
                # If pydub not available, skip unsupported formats
                return None

            # Load and convert audio
            audio = AudioSegment.from_file(str(source_path), format=source_format)

            # Export as WAV to bytes buffer
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format='wav')
            audio_data = wav_buffer.getvalue()
            audio_format = 'wav'
        else:
            # Read audio data directly for supported formats
            with source_path.open("rb") as audio_handle:
                audio_data = audio_handle.read()
            audio_format = source_format

        # Use Chat Completions API with audio input
        import base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        response = client.chat.completions.create(
            model=parsed_config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format
                            }
                        },
                        {
                            "type": "text",
                            "text": parsed_config.prompt or "Transcribe this audio."
                        }
                    ]
                }
            ]
        )

        transcript_text = response.choices[0].message.content or ""

        return ExtractedText(
            text=transcript_text.strip(),
            producer_extractor_id=self.extractor_id
        )
