"""
Analysis backend registry for Biblicus.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import CorpusAnalysisBackend


def available_analysis_backends() -> Dict[str, Type[CorpusAnalysisBackend]]:
    """
    Return the registered analysis backends.

    :return: Mapping of analysis identifiers to backend classes.
    :rtype: dict[str, Type[CorpusAnalysisBackend]]
    """
    from .markov import MarkovBackend
    from .profiling import ProfilingBackend
    from .topic_modeling import TopicModelingBackend

    return {
        ProfilingBackend.analysis_id: ProfilingBackend,
        TopicModelingBackend.analysis_id: TopicModelingBackend,
        MarkovBackend.analysis_id: MarkovBackend,
    }


def get_analysis_backend(analysis_id: str) -> CorpusAnalysisBackend:
    """
    Instantiate an analysis backend by identifier.

    :param analysis_id: Analysis backend identifier.
    :type analysis_id: str
    :return: Analysis backend instance.
    :rtype: CorpusAnalysisBackend
    :raises KeyError: If the analysis backend identifier is unknown.
    """
    registry = available_analysis_backends()
    backend_class = registry.get(analysis_id)
    if backend_class is None:
        known = ", ".join(sorted(registry))
        raise KeyError(f"Unknown analysis retriever '{analysis_id}'. Known retrievers: {known}")
    return backend_class()
