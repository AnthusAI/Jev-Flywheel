"""
Smart override selection extractor that intelligently chooses between extraction results.

This extractor implements the smart override behavior where it compares the most recent
extraction against previous ones and makes an intelligent choice based on content quality
and confidence scores.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class SelectSmartOverrideConfig(BaseModel):
    """
    Configuration for smart override selection.

    :ivar media_type_patterns: List of media type patterns to consider (e.g., image/*).
    :vartype media_type_patterns: list[str]
    :ivar min_confidence_threshold: Minimum confidence to consider an extraction good.
    :vartype min_confidence_threshold: float
    :ivar min_text_length: Minimum text length to consider an extraction meaningful.
    :vartype min_text_length: int
    """

    model_config = ConfigDict(extra="forbid")

    media_type_patterns: List[str] = Field(default_factory=lambda: ["*/*"])
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_text_length: int = Field(default=10, ge=0)


class SelectSmartOverrideExtractor(TextExtractor):
    """
    Smart override selector that intelligently chooses between extraction results.

    This extractor applies smart override logic for items matching the configured media
    type patterns. The selection rules are:

    1. If the item's media type doesn't match any configured patterns, use last extraction.
    2. If the last extraction has meaningful content, use it.
    3. If the last extraction is empty or low-confidence but a previous extraction has
       good content with confidence, use the previous one.
    4. Otherwise, use the last extraction.

    Meaningful content is defined as text length >= min_text_length AND (confidence
    >= min_confidence_threshold OR confidence is not available).

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "select-smart-override"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate selection extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: SelectSmartOverrideConfig
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

        return SelectSmartOverrideConfig.model_validate(parsed_config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Select extracted text using smart override logic.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: SelectSmartOverrideConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Selected extracted text payload or None when no prior outputs exist.
        :rtype: ExtractedText or None
        """
        _ = corpus
        parsed_config = (
            config
            if isinstance(config, SelectSmartOverrideConfig)
            else SelectSmartOverrideConfig.model_validate(config)
        )

        matches_pattern = any(
            fnmatch.fnmatch(item.media_type, pattern)
            for pattern in parsed_config.media_type_patterns
        )

        extracted_candidates = [e for e in previous_extractions if e.text is not None]

        if not extracted_candidates:
            return None

        if not matches_pattern:
            return self._extraction_to_result(extracted_candidates[-1])

        last_extraction = extracted_candidates[-1]
        previous_candidates = extracted_candidates[:-1]

        last_is_meaningful = self._is_meaningful(last_extraction, parsed_config)

        if last_is_meaningful:
            return self._extraction_to_result(last_extraction)

        best_candidate = None
        best_confidence = -1.0
        for prev in previous_candidates:
            if self._is_meaningful(prev, parsed_config):
                prev_confidence = prev.confidence if prev.confidence is not None else 0.0
                if prev_confidence > best_confidence:
                    best_candidate = prev
                    best_confidence = prev_confidence

        if best_candidate is not None:
            return self._extraction_to_result(best_candidate)

        return self._extraction_to_result(last_extraction)

    def _is_meaningful(
        self, extraction: ExtractionStageOutput, config: SelectSmartOverrideConfig
    ) -> bool:
        """
        Check if an extraction has meaningful content.

        :param extraction: Extraction stage output to check.
        :type extraction: ExtractionStageOutput
        :param config: Parsed configuration.
        :type config: SelectSmartOverrideConfig
        :return: True if the extraction has meaningful content.
        :rtype: bool
        """
        text = (extraction.text or "").strip()
        if len(text) < config.min_text_length:
            return False

        confidence = extraction.confidence
        if confidence is not None and confidence < config.min_confidence_threshold:
            return False

        return True

    def _extraction_to_result(self, extraction: ExtractionStageOutput) -> ExtractedText:
        """
        Convert an ExtractionStageOutput to ExtractedText.

        :param extraction: Extraction stage output to convert.
        :type extraction: ExtractionStageOutput
        :return: Extracted text result.
        :rtype: ExtractedText
        """
        producer = extraction.producer_extractor_id or extraction.extractor_id
        return ExtractedText(
            text=extraction.text or "",
            producer_extractor_id=producer,
            source_stage_index=extraction.stage_index,
            confidence=extraction.confidence,
        )
