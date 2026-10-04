"""
Selection extractor that chooses text from previous pipeline outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class SelectTextExtractorConfig(BaseModel):
    """
    Configuration for the selection extractor.

    The selection extractor is intentionally minimal and requires no configuration.
    """

    model_config = ConfigDict(extra="forbid")


class SelectTextExtractor(TextExtractor):
    """
    Extractor plugin that selects from previous pipeline outputs.

    This extractor is used as a final stage when you want to make an explicit choice among
    multiple extraction outputs in the same pipeline.

    It selects the first usable extracted text in pipeline order. Usable means the text is
    non-empty after stripping whitespace. If no usable text exists but prior extracted text
    exists, it selects the first extracted text even if it is empty.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "select-text"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate selection extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: SelectTextExtractorConfig
        """
        return SelectTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Select extracted text from previous pipeline outputs.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: SelectTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Selected extracted text payload or None when no prior outputs exist.
        :rtype: ExtractedText or None
        """
        _ = corpus
        _ = item
        _ = config

        extracted_candidates = [entry for entry in previous_extractions if entry.text is not None]
        usable_candidates = [entry for entry in extracted_candidates if entry.text.strip()]

        if usable_candidates:
            candidate = usable_candidates[0]
            producer = candidate.producer_extractor_id or candidate.extractor_id
            return ExtractedText(
                text=candidate.text or "",
                producer_extractor_id=producer,
                source_stage_index=candidate.stage_index,
            )

        if extracted_candidates:
            candidate = extracted_candidates[0]
            producer = candidate.producer_extractor_id or candidate.extractor_id
            return ExtractedText(
                text=candidate.text or "",
                producer_extractor_id=producer,
                source_stage_index=candidate.stage_index,
            )

        return None
