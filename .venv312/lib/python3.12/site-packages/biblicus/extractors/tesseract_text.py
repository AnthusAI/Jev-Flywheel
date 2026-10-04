"""
Tesseract OCR extractor plugin.

This extractor performs optical character recognition on image items using Tesseract OCR.
It supports layout-aware OCR by reading region metadata from previous pipeline stages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class TesseractExtractorConfig(BaseModel):
    """
    Configuration for the Tesseract OCR extractor.

    :ivar min_confidence: Minimum per-word confidence to include in output (0.0-1.0).
    :vartype min_confidence: float
    :ivar joiner: Joiner used to combine recognized text segments.
    :vartype joiner: str
    :ivar lang: Tesseract language code (e.g., 'eng', 'fra', 'chi_sim').
    :vartype lang: str
    :ivar psm: Page segmentation mode (0-13). Default 3 is fully automatic.
    :vartype psm: int
    :ivar oem: OCR Engine Mode (0=Legacy, 1=LSTM, 2=Combined, 3=Default).
    :vartype oem: int
    :ivar use_layout_metadata: Read region metadata from previous pipeline stage.
    :vartype use_layout_metadata: bool
    """

    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    joiner: str = Field(default="\n")
    lang: str = Field(default="eng")
    psm: int = Field(default=3, ge=0, le=13)
    oem: int = Field(default=3, ge=0, le=3)
    use_layout_metadata: bool = Field(default=False)


class TesseractExtractor(TextExtractor):
    """
    Extractor plugin that performs optical character recognition using Tesseract OCR.

    This extractor can operate in two modes:
    1. Standard mode: OCR the entire image
    2. Layout-aware mode: Read region metadata from previous stage and OCR each region separately

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "ocr-tesseract"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure Tesseract is available.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: TesseractExtractorConfig
        :raises ExtractionSnapshotFatalError: If pytesseract or tesseract is missing.
        """
        try:
            import pytesseract  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Tesseract extractor requires pytesseract. "
                'Install it with pip install "biblicus[tesseract]".'
            ) from import_error

        # Check if tesseract binary is available
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception as check_error:
            raise ExtractionSnapshotFatalError(
                "Tesseract OCR binary not found. "
                "Install tesseract-ocr: brew install tesseract (macOS) or "
                "apt-get install tesseract-ocr (Ubuntu)."
            ) from check_error

        return TesseractExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text from an image item using Tesseract OCR.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item to extract text from.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: TesseractExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[ExtractionStageOutput]
        :return: Extracted text payload or None when item is not an image.
        :rtype: ExtractedText or None
        """
        parsed_config = (
            config
            if isinstance(config, TesseractExtractorConfig)
            else TesseractExtractorConfig.model_validate(config)
        )

        # Skip non-image items
        if not item.media_type.startswith("image/"):
            return None

        # Get image path
        source_path = corpus.root / item.relpath

        # Check for layout metadata if layout-aware mode is enabled
        if parsed_config.use_layout_metadata and previous_extractions:
            layout_metadata = previous_extractions[-1].metadata
            if layout_metadata and "regions" in layout_metadata:
                return self._extract_with_layout(source_path, layout_metadata, parsed_config)

        # Standard mode: OCR entire image
        return self._extract_full_image(source_path, parsed_config)

    def _extract_full_image(self, image_path, config: TesseractExtractorConfig) -> ExtractedText:
        """
        Extract text from entire image using Tesseract.

        :param image_path: Path to image file.
        :type image_path: Path
        :param config: Extractor configuration.
        :type config: TesseractExtractorConfig
        :return: Extracted text with confidence.
        :rtype: ExtractedText
        """
        import pytesseract
        from PIL import Image

        # Build Tesseract configuration string
        tesseract_config = f"--oem {config.oem} --psm {config.psm}"

        # Load image
        image = Image.open(str(image_path))

        # Get detailed OCR data with confidence scores
        data = pytesseract.image_to_data(
            image,
            lang=config.lang,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT,
        )

        # Extract words with confidence filtering
        words = []
        confidences = []

        for i, word_text in enumerate(data["text"]):
            if not word_text.strip():
                continue

            # pytesseract returns confidence as int 0-100, convert to float 0.0-1.0
            confidence = float(data["conf"][i]) / 100.0

            if confidence >= config.min_confidence:
                words.append(word_text)
                confidences.append(confidence)

        # Join words and calculate average confidence
        text = config.joiner.join(words) if words else ""
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return ExtractedText(
            text=text,
            producer_extractor_id=self.extractor_id,
            confidence=avg_confidence,
        )

    def _extract_with_layout(
        self, image_path, layout_metadata: Dict[str, Any], config: TesseractExtractorConfig
    ) -> ExtractedText:
        """
        Extract text from image regions using layout metadata.

        :param image_path: Path to image file.
        :type image_path: Path
        :param layout_metadata: Layout regions and reading order from previous stage.
        :type layout_metadata: dict
        :param config: Extractor configuration.
        :type config: TesseractExtractorConfig
        :return: Extracted text with confidence.
        :rtype: ExtractedText
        """
        import pytesseract
        from PIL import Image

        # Build Tesseract configuration string
        tesseract_config = f"--oem {config.oem} --psm {config.psm}"

        # Load full image
        image = Image.open(str(image_path))

        # Get regions sorted by reading order
        regions = sorted(layout_metadata.get("regions", []), key=lambda r: r.get("order", 999))

        # OCR each region
        region_texts = []
        all_confidences = []

        for region in regions:
            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            # Crop region (bbox format: [x, y, w, h])
            x, y, w, h = bbox
            region_image = image.crop((x, y, x + w, y + h))

            # OCR this region
            data = pytesseract.image_to_data(
                region_image,
                lang=config.lang,
                config=tesseract_config,
                output_type=pytesseract.Output.DICT,
            )

            # Extract words from this region
            region_words = []
            for i, word_text in enumerate(data["text"]):
                if not word_text.strip():
                    continue

                confidence = float(data["conf"][i]) / 100.0
                if confidence >= config.min_confidence:
                    region_words.append(word_text)
                    all_confidences.append(confidence)

            if region_words:
                region_texts.append(config.joiner.join(region_words))

        # Join all region texts
        text = "\n\n".join(region_texts) if region_texts else ""
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else None

        # Include metadata about regions processed
        metadata = {
            "regions_processed": len(regions),
            "regions_with_text": len(region_texts),
        }

        return ExtractedText(
            text=text,
            producer_extractor_id=self.extractor_id,
            confidence=avg_confidence,
            metadata=metadata,
        )
