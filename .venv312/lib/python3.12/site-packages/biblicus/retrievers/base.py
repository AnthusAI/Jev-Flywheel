"""
Retriever interface for Biblicus retrieval engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..corpus import Corpus
from ..models import QueryBudget, RetrievalResult, RetrievalSnapshot


class Retriever(ABC):
    """
    Abstract interface for retrievers.

    :ivar retriever_id: Identifier string for the retriever.
    :vartype retriever_id: str
    """

    retriever_id: str

    @abstractmethod
    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Build or register a retrieval snapshot for the retriever.

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human name for the configuration.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: RetrievalSnapshot
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        corpus: Corpus,
        *,
        snapshot: RetrievalSnapshot,
        query_text: str,
        budget: QueryBudget,
    ) -> RetrievalResult:
        """
        Run a retrieval query against a retriever.

        :param corpus: Corpus associated with the snapshot.
        :type corpus: Corpus
        :param snapshot: Snapshot manifest to use for querying.
        :type snapshot: RetrievalSnapshot
        :param query_text: Query text to execute.
        :type query_text: str
        :param budget: Evidence selection budget.
        :type budget: QueryBudget
        :return: Retrieval results containing evidence.
        :rtype: RetrievalResult
        """
        raise NotImplementedError
