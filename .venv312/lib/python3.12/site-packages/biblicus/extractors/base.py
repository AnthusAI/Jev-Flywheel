"""
Base interfaces for text extraction plugins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..models import CatalogItem, ExtractedText, ExtractionStageOutput

Corpus = Any


class TextExtractor(ABC):
    """
    Abstract interface for plugins that derive text artifacts from corpus items.

    A text extractor is intentionally independent from retrieval backends. It can be swapped
    independently so that different extraction approaches can be evaluated against the same corpus
    and the same retrieval backend.

    :ivar extractor_id: Identifier string for the extractor plugin.
    :vartype extractor_id: str
    """

    extractor_id: str

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate extractor configuration and return a parsed model.

        :param config: Extractor configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration model.
        :rtype: pydantic.BaseModel
        :raises ValueError: If the configuration is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_text(
        self,
        *,
        corpus: "Corpus",
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Derive text for a catalog item.

        Returning None indicates that the item was intentionally skipped.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item to process.
        :type item: CatalogItem
        :param config: Parsed extractor configuration.
        :type config: pydantic.BaseModel
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :return: Extracted text payload or None when skipped.
        :rtype: ExtractedText or None
        """
        raise NotImplementedError
