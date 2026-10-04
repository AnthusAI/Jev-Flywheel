"""
Simple override selection extractor that always uses the last extraction for matching types.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class SelectOverrideConfig(BaseModel):
    """
    Configuration for simple override selection.

    :ivar media_type_patterns: List of media type patterns where override applies.
    :vartype media_type_patterns: list[str]
    :ivar fallback_to_first: If True, fall back to first extraction when no override match.
    :vartype fallback_to_first: bool
    """

    model_config = ConfigDict(extra="forbid")

    media_type_patterns: List[str] = Field(default_factory=lambda: ["*/*"])
    fallback_to_first: bool = Field(default=False)


class SelectOverrideExtractor(TextExtractor):
    """
    Simple override selector that uses the last extraction for matching media types.

    For items matching the configured patterns, always use the last extraction.
    For non-matching items, use the first extraction (if fallback_to_first) or last.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "select-override"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate selection extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: SelectOverrideConfig
        """
        import json

        # Parse JSON values from CLI string format
        parsed_config = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("["):
                try:
                    parsed_config[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_config[key] = value
            else:
                parsed_config[key] = value

        return SelectOverrideConfig.model_validate(parsed_config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Select extracted text using simple override logic.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: SelectOverrideConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Selected extracted text payload or None when no prior outputs exist.
        :rtype: ExtractedText or None
        """
        _ = corpus
        parsed_config = (
            config
            if isinstance(config, SelectOverrideConfig)
            else SelectOverrideConfig.model_validate(config)
        )

        extracted_candidates = [e for e in previous_extractions if e.text is not None]

        if not extracted_candidates:
            return None

        matches_pattern = any(
            fnmatch.fnmatch(item.media_type, pattern)
            for pattern in parsed_config.media_type_patterns
        )

        if matches_pattern:
            candidate = extracted_candidates[-1]
        elif parsed_config.fallback_to_first:
            candidate = extracted_candidates[0]
        else:
            candidate = extracted_candidates[-1]

        producer = candidate.producer_extractor_id or candidate.extractor_id
        return ExtractedText(
            text=candidate.text or "",
            producer_extractor_id=producer,
            source_stage_index=candidate.stage_index,
            confidence=candidate.confidence,
        )
