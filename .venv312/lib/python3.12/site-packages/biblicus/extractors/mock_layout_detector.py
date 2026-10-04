"""
Mock layout detector extractor.

This extractor emits deterministic layout metadata for testing and demos.
It does not perform real detection and is intended for layout-aware OCR pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..corpus import Corpus
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class MockLayoutDetectorConfig(BaseModel):
    """
    Configuration for the mock layout detector.

    :ivar layout_type: Layout preset to use (single-column, two-column, complex).
    :vartype layout_type: str
    """

    model_config = ConfigDict(extra="forbid")

    layout_type: str = Field(default="single-column")

    @model_validator(mode="after")
    def _validate_layout_type(self) -> "MockLayoutDetectorConfig":
        allowed = {"single-column", "two-column", "complex"}
        if self.layout_type not in allowed:
            raise ValueError(
                f"Unsupported layout_type: {self.layout_type}. "
                f"Expected one of: {', '.join(sorted(allowed))}."
            )
        return self


class MockLayoutDetectorExtractor(TextExtractor):
    """
    Extractor plugin that emits deterministic layout metadata.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "mock-layout-detector"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate mock layout detector configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: MockLayoutDetectorConfig
        """
        return MockLayoutDetectorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Emit layout metadata for an image item.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: MockLayoutDetectorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[ExtractionStageOutput]
        :return: ExtractedText with empty text and layout metadata, or None if not an image.
        :rtype: ExtractedText or None
        """
        _ = corpus
        _ = previous_extractions

        if not item.media_type or not item.media_type.startswith("image/"):
            return None

        parsed_config = (
            config
            if isinstance(config, MockLayoutDetectorConfig)
            else MockLayoutDetectorConfig.model_validate(config)
        )

        metadata = _layout_metadata_for(parsed_config.layout_type)

        return ExtractedText(
            text="",
            metadata=metadata,
            producer_extractor_id=self.extractor_id,
        )


def _layout_metadata_for(layout_type: str) -> Dict[str, Any]:
    """
    Return deterministic layout metadata for a layout preset.

    :param layout_type: Layout preset.
    :type layout_type: str
    :return: Layout metadata mapping.
    :rtype: dict[str, Any]
    """
    presets: Dict[str, List[Dict[str, Any]]] = {
        "single-column": [
            {"id": 1, "type": "header", "bbox": [50, 50, 700, 100], "order": 1},
            {"id": 2, "type": "text", "bbox": [50, 120, 700, 800], "order": 2},
        ],
        "two-column": [
            {"id": 1, "type": "header", "bbox": [50, 50, 700, 100], "order": 1},
            {"id": 2, "type": "text", "bbox": [50, 120, 350, 800], "order": 2},
            {"id": 3, "type": "text", "bbox": [380, 120, 350, 800], "order": 3},
        ],
        "complex": [
            {"id": 1, "type": "header", "bbox": [50, 50, 700, 100], "order": 1},
            {"id": 2, "type": "text", "bbox": [50, 120, 350, 600], "order": 2},
            {"id": 3, "type": "text", "bbox": [380, 120, 350, 600], "order": 3},
            {"id": 4, "type": "table", "bbox": [50, 740, 680, 260], "order": 4},
        ],
    }

    regions = presets.get(layout_type, [])

    return {
        "layout_detector": "mock-layout-detector",
        "layout_type": layout_type,
        "regions": regions,
        "num_regions": len(regions),
    }
