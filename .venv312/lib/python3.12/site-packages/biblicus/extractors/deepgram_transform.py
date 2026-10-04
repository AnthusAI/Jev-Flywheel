"""
Deepgram transcription transform extractor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class DeepgramTranscriptTransformConfig(BaseModel):
    """
    Configuration for transforming Deepgram structured metadata into text.

    :ivar source: Which Deepgram representation to render (transcript, utterances, or words).
    :vartype source: str
    :ivar channels: Optional list of channel indices to include.
    :vartype channels: list[int] or None
    :ivar speakers: Optional list of speaker indices to include.
    :vartype speakers: list[int] or None
    :ivar join_with: Separator used when joining utterances or words.
    :vartype join_with: str
    :ivar include_channel_labels: Whether to prefix text with channel labels.
    :vartype include_channel_labels: bool
    :ivar include_speaker_labels: Whether to prefix text with speaker labels.
    :vartype include_speaker_labels: bool
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="transcript", min_length=1)
    channels: Optional[List[int]] = None
    speakers: Optional[List[int]] = None
    join_with: str = Field(default=" ", min_length=1)
    include_channel_labels: bool = False
    include_speaker_labels: bool = False


class DeepgramTranscriptTransformExtractor(TextExtractor):
    """
    Transform Deepgram structured metadata into text output.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "deepgram-transform"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate transform configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: DeepgramTranscriptTransformConfig
        """
        parsed = DeepgramTranscriptTransformConfig.model_validate(config)
        if parsed.source not in {"transcript", "utterances", "words"}:
            raise ValueError(
                "deepgram-transform source must be one of transcript, utterances, or words"
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
        Produce text from Deepgram structured metadata in prior stage outputs.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: DeepgramTranscriptTransformConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload or None when metadata is missing.
        :rtype: ExtractedText or None
        """
        _ = corpus
        if not item.media_type.startswith("audio/"):
            return None

        parsed = (
            config
            if isinstance(config, DeepgramTranscriptTransformConfig)
            else DeepgramTranscriptTransformConfig.model_validate(config)
        )

        deepgram_payload = _find_deepgram_payload(previous_extractions=previous_extractions)
        if deepgram_payload is None:
            raise ValueError("deepgram-transform requires Deepgram metadata from a prior stage")

        text = _render_deepgram_text(payload=deepgram_payload, config=parsed)
        return ExtractedText(text=text, producer_extractor_id=self.extractor_id)


def _find_deepgram_payload(
    *, previous_extractions: List[ExtractionStageOutput]
) -> Optional[Dict[str, Any]]:
    for output in reversed(previous_extractions):
        if not output.metadata:
            continue
        payload = output.metadata.get("deepgram")
        if isinstance(payload, dict):
            return payload
    return None


def _render_deepgram_text(
    *, payload: Dict[str, Any], config: DeepgramTranscriptTransformConfig
) -> str:
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    channel_indices = (
        [index for index in range(len(channels))]
        if config.channels is None
        else list(config.channels)
    )
    chunks: List[str] = []
    for channel_index, channel in enumerate(channels):
        if channel_index not in channel_indices:
            continue
        alternatives = channel.get("alternatives") or []
        if not alternatives:
            continue
        alternative = alternatives[0] or {}
        if config.source == "transcript":
            transcript = str(alternative.get("transcript") or "").strip()
            if transcript:
                chunks.append(_prefix_with_labels(
                    transcript,
                    channel_index=channel_index,
                    speaker=None,
                    config=config,
                ))
            continue
        if config.source == "utterances":
            utterances = alternative.get("utterances") or results.get("utterances") or []
            utterance_texts: List[str] = []
            for entry in utterances:
                if not isinstance(entry, dict):
                    continue
                speaker = entry.get("speaker")
                channel_value = entry.get("channel")
                if config.channels is not None and channel_value is not None:
                    if int(channel_value) not in channel_indices:
                        continue
                if config.speakers is not None and speaker is not None:
                    if int(speaker) not in config.speakers:
                        continue
                text_value = str(entry.get("transcript") or entry.get("text") or "").strip()
                if not text_value:
                    continue
                utterance_texts.append(
                    _prefix_with_labels(
                        text_value,
                        channel_index=channel_index if channel_value is None else int(channel_value),
                        speaker=int(speaker) if speaker is not None else None,
                        config=config,
                    )
                )
            if utterance_texts:
                chunks.append(config.join_with.join(utterance_texts))
            continue
        words = alternative.get("words") or results.get("words") or []
        word_texts: List[str] = []
        current_speaker: Optional[int] = None
        current_channel: Optional[int] = channel_index
        for entry in words:
            if not isinstance(entry, dict):
                continue
            speaker = entry.get("speaker")
            channel_value = entry.get("channel")
            if config.channels is not None and channel_value is not None:
                if int(channel_value) not in channel_indices:
                    continue
            if config.speakers is not None and speaker is not None:
                if int(speaker) not in config.speakers:
                    continue
            word_value = str(entry.get("punctuated_word") or entry.get("word") or "").strip()
            if not word_value:
                continue
            if config.include_speaker_labels and speaker is not None:
                speaker_value = int(speaker)
                channel_value_int = int(channel_value) if channel_value is not None else channel_index
                if word_texts and (speaker_value != current_speaker or channel_value_int != current_channel):
                    chunks.append(
                        _prefix_with_labels(
                            config.join_with.join(word_texts),
                            channel_index=current_channel,
                            speaker=current_speaker,
                            config=config,
                        )
                    )
                    word_texts = []
                current_speaker = speaker_value
                current_channel = channel_value_int
            word_texts.append(word_value)
        if word_texts:
            chunks.append(
                _prefix_with_labels(
                    config.join_with.join(word_texts),
                    channel_index=current_channel if config.include_speaker_labels else channel_index,
                    speaker=current_speaker if config.include_speaker_labels else None,
                    config=config,
                )
            )
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _prefix_with_labels(
    text: str,
    *,
    channel_index: Optional[int],
    speaker: Optional[int],
    config: DeepgramTranscriptTransformConfig,
) -> str:
    labels: List[str] = []
    if config.include_channel_labels and channel_index is not None:
        labels.append(f"Channel {channel_index}")
    if config.include_speaker_labels and speaker is not None:
        labels.append(f"Speaker {speaker}")
    if not labels:
        return text
    return f"{', '.join(labels)}: {text}"
