"""
Pass-through extractor for text items.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..corpus import Corpus
from ..frontmatter import parse_front_matter
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class PassThroughTextExtractorConfig(BaseModel):
    """
    Configuration for the pass-through text extractor.

    This extractor is intentionally minimal and requires no configuration.
    """

    model_config = ConfigDict(extra="forbid")


class PassThroughTextExtractor(TextExtractor):
    """
    Extractor plugin that reads text items from the corpus and returns their text content.

    Non-text items are skipped.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "pass-through-text"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: PassThroughTextExtractorConfig
        """
        return PassThroughTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text by reading the raw item content from the corpus.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: PassThroughTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None if the item is not text.
        :rtype: ExtractedText or None
        """
        _ = config
        _ = previous_extractions
        media_type = item.media_type
        if media_type != "text/markdown" and not media_type.startswith("text/"):
            return None
        raw_bytes = (corpus.root / item.relpath).read_bytes()
        if media_type == "text/markdown":
            markdown_text = raw_bytes.decode("utf-8")
            parsed_document = parse_front_matter(markdown_text)
            return ExtractedText(text=parsed_document.body, producer_extractor_id=self.extractor_id)
        return ExtractedText(
            text=raw_bytes.decode("utf-8"), producer_extractor_id=self.extractor_id
        )
