"""
Metadata-based text extractor plugin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor


class MetadataTextExtractorConfig(BaseModel):
    """
    Configuration for the metadata text extractor.

    The metadata text extractor is intentionally minimal and deterministic.
    It emits a plain text representation derived only from an item's catalog metadata.

    :ivar include_title: Whether to include the item title as the first line, if present.
    :vartype include_title: bool
    :ivar include_tags: Whether to include a ``tags: ...`` line, if tags are present.
    :vartype include_tags: bool
    """

    model_config = ConfigDict(extra="forbid")

    include_title: bool = Field(default=True)
    include_tags: bool = Field(default=True)


class MetadataTextExtractor(TextExtractor):
    """
    Extractor plugin that emits a small, searchable text representation of item metadata.

    The output is intended to be stable and human-readable:

    - If a title exists, the first line is the title.
    - If tags exist, the next line is ``tags: <comma separated tags>``.

    This extractor is useful for:

    - Retrieval over non-text items that carry meaningful metadata.
    - Comparing downstream retrieval backends while holding extraction stable.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "metadata-text"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed config.
        :rtype: MetadataTextExtractorConfig
        """
        return MetadataTextExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus,
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Extract a metadata-based text payload for the item.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: MetadataTextExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload, or ``None`` if no metadata is available.
        :rtype: ExtractedText or None
        """
        parsed_config = (
            config
            if isinstance(config, MetadataTextExtractorConfig)
            else MetadataTextExtractorConfig.model_validate(config)
        )
        _ = corpus
        _ = previous_extractions
        lines: list[str] = []

        if parsed_config.include_title and isinstance(item.title, str) and item.title.strip():
            lines.append(item.title.strip())

        tags = [tag.strip() for tag in item.tags if isinstance(tag, str) and tag.strip()]
        if parsed_config.include_tags and tags:
            lines.append(f"tags: {', '.join(tags)}")

        if not lines:
            return None

        return ExtractedText(text="\n".join(lines), producer_extractor_id=self.extractor_id)
