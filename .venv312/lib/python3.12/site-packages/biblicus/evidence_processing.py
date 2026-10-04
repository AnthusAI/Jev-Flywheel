"""
Evidence processing stages for Biblicus.

Retrieval backends return ranked evidence. Additional stages can be applied without changing the
backend implementation:

- Rerank: reorder evidence.
- Filter: remove evidence.

These stages are explicit so they can be configured, tested, and evaluated independently from the
retrieval backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from .models import Evidence


class EvidenceReranker(ABC):
    """
    Evidence reranker interface.

    :param reranker_id: Stable identifier for this reranker implementation.
    :type reranker_id: str
    """

    reranker_id: str

    @abstractmethod
    def rerank(self, *, query_text: str, evidence: List[Evidence]) -> List[Evidence]:
        """
        Reorder evidence for the given query.

        :param query_text: Query text associated with the evidence.
        :type query_text: str
        :param evidence: Evidence objects to rerank.
        :type evidence: list[Evidence]
        :return: Reranked evidence list.
        :rtype: list[Evidence]
        """


class EvidenceFilter(ABC):
    """
    Evidence filter interface.

    :param filter_id: Stable identifier for this filter implementation.
    :type filter_id: str
    """

    filter_id: str

    @abstractmethod
    def filter(
        self, *, query_text: str, evidence: List[Evidence], config: Dict[str, Any]
    ) -> List[Evidence]:
        """
        Filter evidence for the given query.

        :param query_text: Query text associated with the evidence.
        :type query_text: str
        :param evidence: Evidence objects to filter.
        :type evidence: list[Evidence]
        :param config: Filter-specific configuration values.
        :type config: dict[str, Any]
        :return: Filtered evidence list.
        :rtype: list[Evidence]
        """


class EvidenceRerankLongestText(EvidenceReranker):
    """
    Reranker that prioritizes evidence with longer text.

    This is a deterministic policy that is useful when a downstream context pack is limited by a
    character or token budget and longer evidence is preferred.

    :ivar reranker_id: Stable reranker identifier.
    :vartype reranker_id: str
    """

    reranker_id = "rerank-longest-text"

    def rerank(self, *, query_text: str, evidence: List[Evidence]) -> List[Evidence]:
        """
        Reorder evidence by descending text length.

        :param query_text: Query text associated with the evidence.
        :type query_text: str
        :param evidence: Evidence objects to rerank.
        :type evidence: list[Evidence]
        :return: Evidence list ordered by text length.
        :rtype: list[Evidence]
        """
        return sorted(
            evidence,
            key=lambda evidence_item: (
                -len((evidence_item.text or "").strip()),
                evidence_item.item_id,
            ),
        )


class EvidenceFilterMinimumScoreConfig(BaseModel):
    """
    Configuration for the minimum score evidence filter.

    :ivar minimum_score: Evidence with score below this threshold is removed.
    :vartype minimum_score: float
    """

    model_config = ConfigDict(extra="forbid")

    minimum_score: float = Field(ge=0.0)


class EvidenceFilterMinimumScore(EvidenceFilter):
    """
    Filter that removes evidence below a minimum score threshold.

    :ivar filter_id: Stable filter identifier.
    :vartype filter_id: str
    """

    filter_id = "filter-minimum-score"

    def filter(
        self, *, query_text: str, evidence: List[Evidence], config: Dict[str, Any]
    ) -> List[Evidence]:
        """
        Filter evidence by score threshold.

        :param query_text: Query text associated with the evidence.
        :type query_text: str
        :param evidence: Evidence objects to filter.
        :type evidence: list[Evidence]
        :param config: Filter configuration values.
        :type config: dict[str, Any]
        :return: Evidence list with low-score items removed.
        :rtype: list[Evidence]
        """
        parsed_config = EvidenceFilterMinimumScoreConfig.model_validate(config)
        return [
            evidence_item
            for evidence_item in evidence
            if float(evidence_item.score) >= parsed_config.minimum_score
        ]


_EVIDENCE_RERANKERS: Dict[str, EvidenceReranker] = {
    EvidenceRerankLongestText.reranker_id: EvidenceRerankLongestText(),
}

_EVIDENCE_FILTERS: Dict[str, EvidenceFilter] = {
    EvidenceFilterMinimumScore.filter_id: EvidenceFilterMinimumScore(),
}


def apply_evidence_reranker(
    *, reranker_id: str, query_text: str, evidence: List[Evidence]
) -> List[Evidence]:
    """
    Apply a reranker to evidence by identifier.

    :param reranker_id: Reranker identifier.
    :type reranker_id: str
    :param query_text: Query text associated with the evidence.
    :type query_text: str
    :param evidence: Evidence objects to rerank.
    :type evidence: list[Evidence]
    :return: Reranked evidence list.
    :rtype: list[Evidence]
    :raises KeyError: If the reranker identifier is unknown.
    """
    reranker = _EVIDENCE_RERANKERS[reranker_id]
    return reranker.rerank(query_text=query_text, evidence=evidence)


def apply_evidence_filter(
    *, filter_id: str, query_text: str, evidence: List[Evidence], config: Dict[str, Any]
) -> List[Evidence]:
    """
    Apply a filter to evidence by identifier.

    :param filter_id: Filter identifier.
    :type filter_id: str
    :param query_text: Query text associated with the evidence.
    :type query_text: str
    :param evidence: Evidence objects to filter.
    :type evidence: list[Evidence]
    :param config: Filter-specific configuration values.
    :type config: dict[str, Any]
    :return: Filtered evidence list.
    :rtype: list[Evidence]
    :raises KeyError: If the filter identifier is unknown.
    """
    evidence_filter = _EVIDENCE_FILTERS[filter_id]
    return evidence_filter.filter(query_text=query_text, evidence=evidence, config=config)
