"""
Text extraction plugins for Biblicus.
"""

from __future__ import annotations

from typing import Dict

from .aldea_stt import AldeaSpeechToTextExtractor
from .audio_format_converter import AudioFormatConverterExtractor
from .aws_transcribe_stt import AwsTranscribeSpeechToTextExtractor
from .azure_speech_stt import AzureSpeechToTextExtractor
from .base import TextExtractor
from .deepgram_stt import DeepgramSpeechToTextExtractor
from .deepgram_transform import DeepgramTranscriptTransformExtractor
from .docling_granite_text import DoclingGraniteExtractor
from .docling_smol_text import DoclingSmolExtractor
from .faster_whisper_stt import FasterWhisperSpeechToTextExtractor
from .google_speech_stt import GoogleSpeechToTextExtractor
from .heron_layout import HeronLayoutExtractor
from .markitdown_text import MarkItDownExtractor
from .metadata_text import MetadataTextExtractor
from .mock_layout_detector import MockLayoutDetectorExtractor
from .openai_audio_stt import OpenAiAudioSpeechToTextExtractor
from .openai_stt import OpenAiSpeechToTextExtractor
from .paddleocr_layout import PaddleOCRLayoutExtractor
from .paddleocr_vl_text import PaddleOcrVlExtractor
from .pass_through_text import PassThroughTextExtractor
from .pdf_text import PortableDocumentFormatTextExtractor
from .pipeline import PipelineExtractor
from .rapidocr_text import RapidOcrExtractor
from .select_longest_text import SelectLongestTextExtractor
from .select_override import SelectOverrideExtractor
from .select_smart_override import SelectSmartOverrideExtractor
from .select_text import SelectTextExtractor
from .tesseract_text import TesseractExtractor
from .unstructured_text import UnstructuredExtractor


def get_extractor(extractor_id: str) -> TextExtractor:
    """
    Resolve a built-in text extractor by identifier.

    :param extractor_id: Extractor identifier.
    :type extractor_id: str
    :return: Extractor plugin instance.
    :rtype: TextExtractor
    :raises KeyError: If the extractor identifier is not known.
    """
    extractors: Dict[str, TextExtractor] = {
        MetadataTextExtractor.extractor_id: MetadataTextExtractor(),
        MockLayoutDetectorExtractor.extractor_id: MockLayoutDetectorExtractor(),
        MarkItDownExtractor.extractor_id: MarkItDownExtractor(),
        DoclingSmolExtractor.extractor_id: DoclingSmolExtractor(),
        DoclingGraniteExtractor.extractor_id: DoclingGraniteExtractor(),
        PassThroughTextExtractor.extractor_id: PassThroughTextExtractor(),
        PipelineExtractor.extractor_id: PipelineExtractor(),
        PortableDocumentFormatTextExtractor.extractor_id: PortableDocumentFormatTextExtractor(),
        OpenAiSpeechToTextExtractor.extractor_id: OpenAiSpeechToTextExtractor(),
        OpenAiAudioSpeechToTextExtractor.extractor_id: OpenAiAudioSpeechToTextExtractor(),
        FasterWhisperSpeechToTextExtractor.extractor_id: FasterWhisperSpeechToTextExtractor(),
        AudioFormatConverterExtractor.extractor_id: AudioFormatConverterExtractor(),
        AwsTranscribeSpeechToTextExtractor.extractor_id: AwsTranscribeSpeechToTextExtractor(),
        AzureSpeechToTextExtractor.extractor_id: AzureSpeechToTextExtractor(),
        GoogleSpeechToTextExtractor.extractor_id: GoogleSpeechToTextExtractor(),
        AldeaSpeechToTextExtractor.extractor_id: AldeaSpeechToTextExtractor(),
        DeepgramSpeechToTextExtractor.extractor_id: DeepgramSpeechToTextExtractor(),
        DeepgramTranscriptTransformExtractor.extractor_id: DeepgramTranscriptTransformExtractor(),
        RapidOcrExtractor.extractor_id: RapidOcrExtractor(),
        HeronLayoutExtractor.extractor_id: HeronLayoutExtractor(),
        PaddleOCRLayoutExtractor.extractor_id: PaddleOCRLayoutExtractor(),
        PaddleOcrVlExtractor.extractor_id: PaddleOcrVlExtractor(),
        TesseractExtractor.extractor_id: TesseractExtractor(),
        SelectTextExtractor.extractor_id: SelectTextExtractor(),
        SelectLongestTextExtractor.extractor_id: SelectLongestTextExtractor(),
        SelectSmartOverrideExtractor.extractor_id: SelectSmartOverrideExtractor(),
        SelectOverrideExtractor.extractor_id: SelectOverrideExtractor(),
        UnstructuredExtractor.extractor_id: UnstructuredExtractor(),
    }
    if extractor_id not in extractors:
        raise KeyError(f"Unknown extractor: {extractor_id!r}")
    return extractors[extractor_id]
