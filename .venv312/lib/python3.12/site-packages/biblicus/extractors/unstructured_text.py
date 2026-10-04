"""
Unstructured-based text extraction plugin.

This extractor is implemented as an optional dependency so the core installation stays small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class UnstructuredExtractorConfig(BaseModel):
    """
    Configuration for the Unstructured extractor.

    Version zero does not expose any configuration for this extractor.
    """

    model_config = ConfigDict(extra="forbid")


class UnstructuredExtractor(TextExtractor):
    """
    Extractor plugin backed by the `unstructured` library.

    The intent is broad format coverage as a last-resort extractor. This extractor skips items
    that are already text so the pass-through extractor remains the canonical choice for text
    items and Markdown front matter handling.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "unstructured"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure the dependency is installed.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: UnstructuredExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is not installed.
        """
        try:
            from unstructured.partition.auto import partition  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "Unstructured extractor requires an optional dependency. "
                'Install it with pip install "biblicus[unstructured]".'
            ) from import_error
        return UnstructuredExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text for a non-text item using Unstructured.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: UnstructuredExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is already text.
        :rtype: ExtractedText or None
        """
        _ = config
        _ = previous_extractions
        media_type = item.media_type
        if media_type == "text/markdown" or media_type.startswith("text/"):
            return None

        from unstructured.partition.auto import partition

        source_path = corpus.root / item.relpath
        elements = partition(filename=str(source_path))
        lines: list[str] = []
        for element in elements or []:
            text = getattr(element, "text", None)
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
        combined_text = "\n".join(lines).strip()
        return ExtractedText(text=combined_text, producer_extractor_id=self.extractor_id)
