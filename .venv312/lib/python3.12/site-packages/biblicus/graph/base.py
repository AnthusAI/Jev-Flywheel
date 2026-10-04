"""
Graph extractor interface for Biblicus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel

from ..corpus import Corpus
from ..models import CatalogItem
from .models import GraphExtractionResult


class GraphExtractor(ABC):
    """
    Abstract interface for graph extractors.

    :ivar extractor_id: Identifier string for the graph extractor.
    :vartype extractor_id: str
    """

    extractor_id: str

    @abstractmethod
    def validate_config(self, config: Dict[str, object]) -> BaseModel:
        """
        Validate extractor configuration using Pydantic.

        :param config: Extractor configuration mapping.
        :type config: dict[str, object]
        :return: Parsed configuration model.
        :rtype: pydantic.BaseModel
        """
        raise NotImplementedError

    @abstractmethod
    def extract_graph(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        extracted_text: str,
        config: BaseModel,
    ) -> GraphExtractionResult:
        """
        Extract graph nodes and edges for a single item.

        :param corpus: Corpus containing the item.
        :type corpus: biblicus.corpus.Corpus
        :param item: Catalog item being processed.
        :type item: biblicus.models.CatalogItem
        :param extracted_text: Extracted text for the item.
        :type extracted_text: str
        :param config: Parsed extractor configuration.
        :type config: pydantic.BaseModel
        :return: Graph extraction result.
        :rtype: GraphExtractionResult
        """
        raise NotImplementedError
