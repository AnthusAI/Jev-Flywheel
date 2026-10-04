"""
Selection extractor that chooses the longest available text from previous pipeline outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class SelectLongestTextExtractorConfig(BaseModel):
    """
    Configuration for the longest text selection extractor.

    Version zero does not expose configuration for this extractor.
    """

    model_config = ConfigDict(extra="forbid")


class SelectLongestTextExtractor(TextExtractor):
    """
    Extractor plugin that selects the longest text from previous pipeline outputs.

    This extractor does not attempt to score semantic quality. It is a deterministic
    selection policy for cases where multiple stages can produce usable text for the
    same item.

    The selection rules are:

    - If any prior extracted texts are non-empty after stripping whitespace, choose the one
      with the greatest stripped character count.
    - Ties are broken by earliest pipeline stage index.
    - If no prior extracted texts are usable but prior extracted texts exist, select the
      earliest extracted text even if it is empty.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "select-longest-text"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate selection extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: SelectLongestTextExtractorConfig
        """
        return SelectLongestTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Select the longest extracted text from previous pipeline outputs.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: SelectLongestTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Selected extracted text payload or None when no prior outputs exist.
        :rtype: ExtractedText or None
        """
        _ = corpus
        _ = item
        _ = config

        extracted_candidates = [entry for entry in previous_extractions if entry.text is not None]
        if not extracted_candidates:
            return None

        usable_candidates = [entry for entry in extracted_candidates if entry.text.strip()]
        if usable_candidates:
            candidate = max(usable_candidates, key=lambda entry: len(entry.text.strip()))
            ties = [
                entry
                for entry in usable_candidates
                if len(entry.text.strip()) == len(candidate.text.strip())
            ]
            candidate = min(ties, key=lambda entry: int(entry.stage_index))
        else:
            candidate = min(extracted_candidates, key=lambda entry: int(entry.stage_index))

        producer = candidate.producer_extractor_id or candidate.extractor_id
        return ExtractedText(
            text=candidate.text or "",
            producer_extractor_id=producer,
            source_stage_index=candidate.stage_index,
        )
