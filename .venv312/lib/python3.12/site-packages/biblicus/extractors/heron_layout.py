"""
IBM Heron layout detection extractor.

This extractor uses IBM Research's Heron models for document layout analysis,
as described in arXiv:2509.11720. These are the state-of-the-art models
released September 2025 for the Docling project.

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


class HeronLayoutConfig(BaseModel):
    """
    Configuration for Heron layout detection.

    :ivar model_variant: Which Heron model to use ("base" or "101"). Default: "101".
    :vartype model_variant: str
    :ivar confidence_threshold: Minimum confidence score for detected regions (0.0-1.0). Default: 0.6.
    :vartype confidence_threshold: float
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    model_variant: str = Field(default="101")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class HeronLayoutExtractor(TextExtractor):
    """
    Layout detection extractor using IBM Heron models.

    This extractor analyzes document layout using IBM Research's Heron models
    (RT-DETR V2 based) and identifies regions (text blocks, tables, figures)
    along with their reading order. It outputs metadata that can be consumed by
    OCR extractors with layout-aware capabilities (like Tesseract with
    use_layout_metadata=True).

    Models:
    - heron-101: 76.7M parameters, 78% mAP, most accurate
    - heron-base: 42.9M parameters, faster, good accuracy

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "heron-layout"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure dependencies are installed.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: HeronLayoutConfig
        :raises ExtractionSnapshotFatalError: If required dependencies are not installed.
        """
        parsed = HeronLayoutConfig.model_validate(config)

        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                RTDetrImageProcessor,
                RTDetrV2ForObjectDetection,
            )
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Heron layout extractor requires transformers and torch. "
                'Install with: pip install "transformers>=4.40.0" "torch>=2.0.0"'
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
        :type config: HeronLayoutConfig
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
            if isinstance(config, HeronLayoutConfig)
            else HeronLayoutConfig.model_validate(config)
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
        self, source_path: Path, config: HeronLayoutConfig
    ) -> Dict[str, Any]:
        """
        Detect document layout using IBM Heron models.

        :param source_path: Path to the source image.
        :type source_path: pathlib.Path
        :param config: Parsed configuration.
        :type config: HeronLayoutConfig
        :return: Layout metadata with regions and reading order.
        :rtype: dict
        """
        import torch
        from PIL import Image
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        # Select model based on variant
        if config.model_variant == "101":
            model_name = "ds4sd/docling-layout-heron-101"
        else:
            model_name = "ds4sd/docling-layout-heron"

        # Load model and processor
        # Note: First run will download models (~150MB for heron-101)
        image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        model = RTDetrV2ForObjectDetection.from_pretrained(model_name)

        # Read image
        image = Image.open(source_path).convert("RGB")
        if image is None:
            raise ValueError(f"Failed to read image: {source_path}")

        # Prepare inputs
        inputs = image_processor(images=[image], return_tensors="pt")

        # Detect layout
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=config.confidence_threshold
        )

        # Extract regions from results
        regions = []
        if results:
            result = results[0]
            boxes = result.get("boxes", [])
            scores = result.get("scores", [])
            labels = result.get("labels", [])

            # Get label names from model config
            id2label = model.config.id2label

            # Sort by reading order (top to bottom, left to right)
            detections = list(zip(boxes, scores, labels))
            detections.sort(key=lambda x: (x[0][1].item(), x[0][0].item()))  # Sort by y, then x

            for idx, (box, score, label_id) in enumerate(detections):
                # Convert box tensor to list [x1, y1, x2, y2]
                bbox = [
                    float(box[0].item()),
                    float(box[1].item()),
                    float(box[2].item()),
                    float(box[3].item())
                ]

                # Get label name
                label_name = id2label.get(int(label_id.item()), "unknown")

                regions.append({
                    "id": idx + 1,
                    "type": label_name,
                    "bbox": bbox,
                    "score": float(score.item()),
                    "order": idx + 1,
                })

        return {
            "layout_detector": f"heron-{config.model_variant}",
            "layout_type": "auto-detected",
            "regions": regions,
            "num_regions": len(regions),
            "confidence_threshold": config.confidence_threshold,
        }
