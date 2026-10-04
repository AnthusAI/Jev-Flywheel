"""
Retriever registry for Biblicus retrieval engines.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Retriever
from .embedding_index_file import EmbeddingIndexFileRetriever
from .embedding_index_inmemory import EmbeddingIndexInMemoryRetriever
from .hybrid import HybridRetriever
from .scan import ScanRetriever
from .sqlite_full_text_search import SqliteFullTextSearchRetriever
from .tf_vector import TfVectorRetriever


def available_retrievers() -> Dict[str, Type[Retriever]]:
    """
    Return the registered retrievers.

    :return: Mapping of retriever identifiers to retriever classes.
    :rtype: dict[str, Type[Retriever]]
    """
    return {
        EmbeddingIndexFileRetriever.retriever_id: EmbeddingIndexFileRetriever,
        EmbeddingIndexInMemoryRetriever.retriever_id: EmbeddingIndexInMemoryRetriever,
        HybridRetriever.retriever_id: HybridRetriever,
        ScanRetriever.retriever_id: ScanRetriever,
        SqliteFullTextSearchRetriever.retriever_id: SqliteFullTextSearchRetriever,
        TfVectorRetriever.retriever_id: TfVectorRetriever,
    }


def get_retriever(retriever_id: str) -> Retriever:
    """
    Instantiate a retriever by identifier.

    :param retriever_id: Retriever identifier.
    :type retriever_id: str
    :return: Retriever instance.
    :rtype: Retriever
    :raises KeyError: If the retriever identifier is unknown.
    """
    registry = available_retrievers()
    retriever_class = registry.get(retriever_id)
    if retriever_class is None:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown retriever '{retriever_id}'. Known retrievers: {known}")
    return retriever_class()
