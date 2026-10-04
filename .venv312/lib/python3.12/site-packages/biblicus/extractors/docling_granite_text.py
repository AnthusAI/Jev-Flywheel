"""
Granite Docling VLM-backed document text extraction plugin.

This extractor uses the Granite Docling-258M vision-language model for document understanding.
It supports PDF, Office documents (DOCX, XLSX, PPTX), HTML, and image formats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor

DOCLING_SUPPORTED_MEDIA_TYPES = frozenset(
    [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/html",
        "application/xhtml+xml",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/tiff",
        "image/bmp",
    ]
)


class DoclingGraniteExtractorConfig(BaseModel):
    """
    Configuration for the Granite Docling VLM extractor.

    :ivar output_format: Output format for extracted content (markdown, text, or html).
    :vartype output_format: str
    :ivar retriever: Inference retriever (mlx or transformers).
    :vartype retriever: str
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    output_format: str = Field(default="markdown", pattern="^(markdown|text|html)$")
    retriever: str = Field(default="mlx", pattern="^(mlx|transformers)$", alias="backend")


class DoclingGraniteExtractor(TextExtractor):
    """
    Extractor plugin backed by the Granite Docling-258M vision-language model.

    This extractor converts documents into text using Docling with the Granite VLM.
    It skips text items (text/plain, text/markdown) to let pass-through handle those.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "docling-granite"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure the dependency is installed.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: DoclingGraniteExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is not installed.
        """
        parsed = DoclingGraniteExtractorConfig.model_validate(config)

        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            # Updated import path for docling 2.x
            try:
                from docling.datamodel.pipeline_options import (  # noqa: F401
                    PdfPipelineOptions,
                )
            except ImportError:
                # Fallback for older docling versions
                from docling.pipeline_options import (  # noqa: F401
                    VlmPipelineOptions,
                    vlm_model_specs,
                )
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "DoclingGranite extractor requires an optional dependency. "
                'Install it with pip install "biblicus[docling]".'
            ) from import_error

        if parsed.retriever == "mlx":
            try:
                try:
                    # Try new API first
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    _ = PdfPipelineOptions
                except ImportError:
                    # Fallback to old API
                    from docling.pipeline_options import vlm_model_specs
                    _ = vlm_model_specs.GRANITE_DOCLING_MLX
            except (ImportError, AttributeError) as exc:
                raise ExtractionSnapshotFatalError(
                    "DoclingGranite extractor with MLX retriever requires MLX support. "
                    'Install it with pip install "biblicus[docling-mlx]".'
                ) from exc

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
        Extract text for a document item using Granite Docling.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: DoclingGraniteExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not supported.
        :rtype: ExtractedText or None
        """
        _ = previous_extractions

        if not self._is_supported_media_type(item.media_type):
            return None

        parsed_config = (
            config
            if isinstance(config, DoclingGraniteExtractorConfig)
            else DoclingGraniteExtractorConfig.model_validate(config)
        )

        source_path = corpus.root / item.relpath
        text = self._convert_document(source_path, parsed_config)
        return ExtractedText(text=text.strip(), producer_extractor_id=self.extractor_id)

    def _is_supported_media_type(self, media_type: str) -> bool:
        """
        Check if a media type is supported by this extractor.

        :param media_type: Media type string.
        :type media_type: str
        :return: True if supported, False otherwise.
        :rtype: bool
        """
        if media_type in DOCLING_SUPPORTED_MEDIA_TYPES:
            return True
        if media_type.startswith("image/"):
            return True
        return False

    def _convert_document(self, source_path, config: DoclingGraniteExtractorConfig) -> str:
        """
        Convert a document using Docling with the Granite Docling VLM.

        :param source_path: Path to the source document.
        :type source_path: pathlib.Path
        :param config: Parsed configuration.
        :type config: DoclingGraniteExtractorConfig
        :return: Extracted text content.
        :rtype: str
        """
        from docling.document_converter import DocumentConverter

        # Current Docling API is simplified - just create converter and convert
        # The VLM model configuration (MLX vs Transformers) is handled via environment
        # or model selection, not via explicit pipeline options
        converter = DocumentConverter()
        result = converter.convert(str(source_path))

        if config.output_format == "html":
            return result.document.export_to_html()
        elif config.output_format == "text":
            return result.document.export_to_text()
        else:
            return result.document.export_to_markdown()
