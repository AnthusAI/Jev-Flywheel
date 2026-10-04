"""
Portable Document Format text extractor plugin.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class PortableDocumentFormatTextExtractorConfig(BaseModel):
    """
    Configuration for Portable Document Format text extraction.

    :ivar max_pages: Optional maximum number of pages to process.
    :vartype max_pages: int or None
    """

    model_config = ConfigDict(extra="forbid")

    max_pages: Optional[int] = Field(default=None, ge=1)


class PortableDocumentFormatTextExtractor(TextExtractor):
    """
    Extractor plugin that attempts to extract text from Portable Document Format items.

    This extractor only handles items whose media type is `application/pdf`.
    Items of other media types are skipped.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "pdf-text"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: PortableDocumentFormatTextExtractorConfig
        """
        return PortableDocumentFormatTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text for a Portable Document Format item.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: PortableDocumentFormatTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is not a Portable Document Format item.
        :rtype: ExtractedText or None
        """
        if item.media_type != "application/pdf":
            return None

        _ = previous_extractions
        parsed_config = (
            config
            if isinstance(config, PortableDocumentFormatTextExtractorConfig)
            else PortableDocumentFormatTextExtractorConfig.model_validate(config)
        )

        pdf_path = corpus.root / item.relpath
        pdf_bytes = pdf_path.read_bytes()
        reader = PdfReader(BytesIO(pdf_bytes))

        texts: list[str] = []
        pages = list(reader.pages)
        if parsed_config.max_pages is not None:
            pages = pages[: int(parsed_config.max_pages)]

        for page in pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)

        combined_text = "\n".join(texts).strip()
        return ExtractedText(text=combined_text, producer_extractor_id=self.extractor_id)
