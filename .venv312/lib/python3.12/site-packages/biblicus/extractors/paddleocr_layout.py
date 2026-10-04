"""
PaddleOCR PP-Structure layout detection extractor.

This extractor uses PaddleOCR's PP-Structure model to detect document layout,
including text regions, tables, figures, and their reading order. It outputs
layout metadata that can be consumed by downstream OCR extractors like Tesseract.

Implements a two-stage workflow: detect document layout and reading order,
then pass region metadata to downstream OCR extractors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class PaddleOCRLayoutConfig(BaseModel):
    """
    Configuration for PaddleOCR PP-Structure layout detection.

    :ivar lang: Language for the layout model (default: "en").
    :vartype lang: str
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lang: str = Field(default="en")


class PaddleOCRLayoutExtractor(TextExtractor):
    """
    Layout detection extractor using PaddleOCR PP-Structure.

    This extractor analyzes document layout and identifies regions (text blocks,
    tables, figures) along with their reading order. It outputs metadata that can
    be consumed by OCR extractors with layout-aware capabilities (like Tesseract
    with use_layout_metadata=True).

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "paddleocr-layout"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure PaddleOCR is installed.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: PaddleOCRLayoutConfig
        :raises ExtractionSnapshotFatalError: If PaddleOCR is not installed.
        """
        parsed = PaddleOCRLayoutConfig.model_validate(config)

        try:
            from paddleocr import PPStructureV3  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "PaddleOCR layout extractor requires PaddleOCR. "
                'Install it with pip install "paddleocr" and "paddlex[ocr]".'
            ) from import_error

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
        Extract layout information from a document image.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: PaddleOCRLayoutConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: ExtractedText with empty text but layout metadata, or None if not an image.
        :rtype: ExtractedText or None
        """
        _ = previous_extractions

        # Only process image files
        if not item.media_type or not item.media_type.startswith("image/"):
            return None

        parsed_config = (
            config
            if isinstance(config, PaddleOCRLayoutConfig)
            else PaddleOCRLayoutConfig.model_validate(config)
        )

        source_path = corpus.root / item.relpath

        # Detect layout
        layout_result = self._detect_layout(source_path, parsed_config)

        # Return empty text with layout metadata
        return ExtractedText(
            text="",  # Layout detection doesn't produce text, only metadata
            metadata=layout_result,
            producer_extractor_id=self.extractor_id,
        )

    def _detect_layout(
        self, source_path: Path, config: PaddleOCRLayoutConfig
    ) -> Dict[str, Any]:
        """
        Detect document layout using PaddleOCR PP-Structure.

        :param source_path: Path to the source image.
        :type source_path: pathlib.Path
        :param config: Parsed configuration.
        :type config: PaddleOCRLayoutConfig
        :return: Layout metadata with regions and reading order.
        :rtype: dict
        """
        import os
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

        import cv2
        from paddleocr import PPStructureV3

        # Initialize PP-Structure with layout detection
        # Note: First run will download models (~100MB)
        structure_engine = PPStructureV3(lang=config.lang)

        # Read image
        img = cv2.imread(str(source_path))
        if img is None:
            raise ValueError(f"Failed to read image: {source_path}")

        # Detect layout using predict() method
        result = structure_engine.predict(str(source_path))

        # Extract layout regions from PP-Structure result
        regions = []

        # PP-Structure V3 returns a list with page results
        if isinstance(result, list) and result:
            # Get first page result
            page_result = result[0]

            if isinstance(page_result, dict):
                # Get layout detection results
                layout_det_res = page_result.get('layout_det_res', {})
                boxes = layout_det_res.get('boxes', [])

                # Convert boxes to our region format
                for idx, box in enumerate(boxes):
                    coordinate = box.get('coordinate', [])
                    # Convert coordinate strings to floats and format as [x1, y1, x2, y2]
                    if coordinate and len(coordinate) >= 4:
                        bbox = [float(coordinate[0]), float(coordinate[1]),
                                float(coordinate[2]), float(coordinate[3])]
                    else:
                        bbox = []

                    regions.append({
                        "id": idx + 1,
                        "type": box.get('label', 'text'),
                        "bbox": bbox,
                        "score": box.get('score', 0.0),
                        "order": idx + 1,  # Boxes are already in reading order
                    })

        return {
            "layout_detector": "paddleocr-pp-structurev3",
            "layout_type": "auto-detected",
            "regions": regions,
            "num_regions": len(regions),
        }

    def _parse_region(self, region: Dict[str, Any], order: int) -> Dict[str, Any]:
        """
        Parse a region from PP-Structure result into our metadata format.

        :param region: Region data from PP-Structure.
        :type region: dict
        :param order: Reading order index.
        :type order: int
        :return: Standardized region metadata.
        :rtype: dict
        """
        # PP-Structure regions have: bbox, type/label, and possibly text
        bbox = region.get("bbox", region.get("box", []))
        region_type = region.get("type", region.get("label", "text"))

        return {
            "id": order,
            "type": region_type,
            "bbox": bbox,  # [x1, y1, x2, y2] or [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            "order": order,
        }
