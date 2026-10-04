"""
MarkItDown-based text extraction plugin.

This extractor depends on an optional library so the core installation stays small.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..corpus import Corpus
from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class MarkItDownExtractorConfig(BaseModel):
    """
    Configuration for the MarkItDown extractor.

    :ivar enable_plugins: Whether to enable MarkItDown plugins.
    :vartype enable_plugins: bool
    """

    model_config = ConfigDict(extra="forbid")

    enable_plugins: bool = Field(default=False)


class MarkItDownExtractor(TextExtractor):
    """
    Extractor plugin backed by the `markitdown` library.

    This extractor converts non-text items into Markdown-like text. It skips text items so
    the pass-through extractor remains the canonical choice for text inputs and Markdown
    front matter handling.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "markitdown"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and ensure the dependency is installed.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: MarkItDownExtractorConfig
        :raises ExtractionSnapshotFatalError: If the optional dependency is not installed.
        """
        try:
            import markitdown
            from markitdown import MarkItDown  # noqa: F401
        except ImportError as import_error:
            raise ExtractionSnapshotFatalError(
                "MarkItDown extractor requires an optional dependency. "
                'Install it with pip install "biblicus[markitdown]".'
            ) from import_error
        if sys.version_info < (3, 10) and not getattr(markitdown, "__biblicus_fake__", False):
            raise ExtractionSnapshotFatalError(
                "MarkItDown requires Python 3.10 or higher. "
                "Upgrade your interpreter or use a compatible extractor."
            )
        return MarkItDownExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract text for a non-text item using MarkItDown.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: MarkItDownExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or None when the item is already text.
        :rtype: ExtractedText or None
        """
        parsed_config = (
            config
            if isinstance(config, MarkItDownExtractorConfig)
            else MarkItDownExtractorConfig.model_validate(config)
        )
        _ = previous_extractions
        media_type = item.media_type
        if media_type == "text/markdown" or media_type.startswith("text/"):
            return None

        from markitdown import MarkItDown

        source_path = corpus.root / item.relpath
        converter = MarkItDown(enable_plugins=parsed_config.enable_plugins)
        conversion_result = converter.convert(str(source_path))
        extracted_text = _resolve_markitdown_text(conversion_result).strip()
        return ExtractedText(text=extracted_text, producer_extractor_id=self.extractor_id)


def _resolve_markitdown_text(conversion_result: object) -> str:
    """
    Resolve a text payload from a MarkItDown conversion result.

    :param conversion_result: Result returned by the MarkItDown converter.
    :type conversion_result: object
    :return: Extracted text payload or an empty string.
    :rtype: str
    """
    if isinstance(conversion_result, str):
        return conversion_result
    if conversion_result is None:
        return ""
    text_content = getattr(conversion_result, "text_content", None)
    if isinstance(text_content, str):
        return text_content
    return ""
