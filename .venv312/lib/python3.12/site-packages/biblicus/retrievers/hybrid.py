"""
Hybrid retriever combining lexical and vector results.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..corpus import Corpus
from ..models import Evidence, QueryBudget, RetrievalResult, RetrievalSnapshot
from ..retrieval import apply_budget, create_configuration_manifest, create_snapshot_manifest
from ..time import utc_now_iso


class HybridConfiguration(BaseModel):
    """
    Configuration for hybrid retrieval fusion.

    :ivar lexical_retriever: Retriever identifier for lexical retrieval.
    :vartype lexical_retriever: str
    :ivar embedding_retriever: Retriever identifier for embedding retrieval.
    :vartype embedding_retriever: str
    :ivar lexical_weight: Weight for lexical scores.
    :vartype lexical_weight: float
    :ivar embedding_weight: Weight for embedding scores.
    :vartype embedding_weight: float
    :ivar lexical_configuration: Optional lexical retriever configuration.
    :vartype lexical_configuration: dict[str, object]
    :ivar embedding_configuration: Optional embedding retriever configuration.
    :vartype embedding_configuration: dict[str, object]
    """

    model_config = ConfigDict(extra="forbid")

    lexical_retriever: str = Field(default="sqlite-full-text-search", min_length=1)
    embedding_retriever: str = Field(default="tf-vector", min_length=1)
    lexical_weight: float = Field(default=0.5, ge=0, le=1)
    embedding_weight: float = Field(default=0.5, ge=0, le=1)
    lexical_configuration: Dict[str, object] = Field(default_factory=dict)
    embedding_configuration: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_weights(self) -> "HybridConfiguration":
        if abs((self.lexical_weight + self.embedding_weight) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1")
        return self


class HybridRetriever:
    """
    Hybrid retriever that fuses lexical and embedding retrieval.

    :ivar retriever_id: Retriever identifier.
    :vartype retriever_id: str
    """

    retriever_id = "hybrid"

    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Build or register a hybrid retrieval snapshot.

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: RetrievalSnapshot
        """
        parsed_config = HybridConfiguration.model_validate(configuration)
        _ensure_retriever_supported(parsed_config)
        lexical_retriever = _resolve_retriever(parsed_config.lexical_retriever)
        embedding_retriever = _resolve_retriever(parsed_config.embedding_retriever)
        lexical_snapshot = lexical_retriever.build_snapshot(
            corpus,
            configuration_name=f"{configuration_name}-lexical",
            configuration=parsed_config.lexical_configuration,
        )
        embedding_snapshot = embedding_retriever.build_snapshot(
            corpus,
            configuration_name=f"{configuration_name}-embedding",
            configuration=parsed_config.embedding_configuration,
        )
        configuration_manifest = create_configuration_manifest(
            retriever_id=self.retriever_id,
            name=configuration_name,
            configuration=parsed_config.model_dump(),
        )
        stats = {
            "lexical_snapshot_id": lexical_snapshot.snapshot_id,
            "embedding_snapshot_id": embedding_snapshot.snapshot_id,
        }
        snapshot = create_snapshot_manifest(
            corpus,
            configuration=configuration_manifest,
            stats=stats,
            snapshot_artifacts=[],
        )
        corpus.write_snapshot(snapshot)
        return snapshot

    def query(
        self,
        corpus: Corpus,
        *,
        snapshot: RetrievalSnapshot,
        query_text: str,
        budget: QueryBudget,
    ) -> RetrievalResult:
        """
        Query using both lexical and embedding retrievers and fuse scores.

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
        configuration = HybridConfiguration.model_validate(snapshot.configuration.configuration)
        _ensure_retriever_supported(configuration)
        lexical_retriever = _resolve_retriever(configuration.lexical_retriever)
        embedding_retriever = _resolve_retriever(configuration.embedding_retriever)
        lexical_snapshot_id = snapshot.stats.get("lexical_snapshot_id")
        embedding_snapshot_id = snapshot.stats.get("embedding_snapshot_id")
        if not lexical_snapshot_id or not embedding_snapshot_id:
            raise ValueError("Hybrid snapshot missing lexical or embedding snapshot identifiers")
        lexical_snapshot = corpus.load_snapshot(str(lexical_snapshot_id))
        embedding_snapshot = corpus.load_snapshot(str(embedding_snapshot_id))
        component_budget = _expand_component_budget(budget)
        lexical_result = lexical_retriever.query(
            corpus, snapshot=lexical_snapshot, query_text=query_text, budget=component_budget
        )
        embedding_result = embedding_retriever.query(
            corpus, snapshot=embedding_snapshot, query_text=query_text, budget=component_budget
        )
        candidates = _fuse_evidence(
            lexical_result.evidence,
            embedding_result.evidence,
            lexical_weight=configuration.lexical_weight,
            embedding_weight=configuration.embedding_weight,
        )
        sorted_candidates = sorted(
            candidates,
            key=lambda evidence_item: (-evidence_item.score, evidence_item.item_id),
        )
        ranked = [
            evidence_item.model_copy(
                update={
                    "rank": index,
                    "configuration_id": snapshot.configuration.configuration_id,
                    "snapshot_id": snapshot.snapshot_id,
                }
            )
            for index, evidence_item in enumerate(sorted_candidates, start=1)
        ]
        evidence = apply_budget(ranked, budget)
        stats = {
            "candidates": len(sorted_candidates),
            "returned": len(evidence),
            "fusion_weights": {
                "lexical": configuration.lexical_weight,
                "embedding": configuration.embedding_weight,
            },
        }
        return RetrievalResult(
            query_text=query_text,
            budget=budget,
            snapshot_id=snapshot.snapshot_id,
            configuration_id=snapshot.configuration.configuration_id,
            retriever_id=snapshot.configuration.retriever_id,
            generated_at=utc_now_iso(),
            evidence=evidence,
            stats=stats,
        )


def _ensure_retriever_supported(configuration: HybridConfiguration) -> None:
    """
    Validate that hybrid retrievers do not reference the hybrid retriever itself.

    :param configuration: Parsed hybrid configuration.
    :type configuration: HybridConfiguration
    :return: None.
    :rtype: None
    :raises ValueError: If hybrid is used as a component retriever.
    """
    if configuration.lexical_retriever == HybridRetriever.retriever_id:
        raise ValueError("Hybrid retriever cannot use itself as the lexical retriever")
    if configuration.embedding_retriever == HybridRetriever.retriever_id:
        raise ValueError("Hybrid retriever cannot use itself as the embedding retriever")


def _resolve_retriever(retriever_id: str):
    """
    Resolve a retriever by identifier.

    :param retriever_id: Retriever identifier.
    :type retriever_id: str
    :return: Retriever instance.
    :rtype: object
    """
    from biblicus.retrievers import get_retriever  # Delayed import to avoid circular import

    return get_retriever(retriever_id)


def _expand_component_budget(budget: QueryBudget, *, multiplier: int = 5) -> QueryBudget:
    """
    Expand a final budget to collect more candidates for fusion.

    :param budget: Final evidence budget.
    :type budget: QueryBudget
    :param multiplier: Candidate expansion multiplier.
    :type multiplier: int
    :return: Expanded budget for component retrievers.
    :rtype: QueryBudget
    """
    maximum_total_characters = budget.maximum_total_characters
    expanded_characters = (
        maximum_total_characters * multiplier if maximum_total_characters is not None else None
    )
    expanded_max_items_per_source = (
        budget.max_items_per_source * multiplier
        if budget.max_items_per_source is not None
        else None
    )
    requested_items = budget.max_total_items + budget.offset
    return QueryBudget(
        max_total_items=requested_items * multiplier,
        offset=0,
        maximum_total_characters=expanded_characters,
        max_items_per_source=expanded_max_items_per_source,
    )


def _fuse_evidence(
    lexical: List[Evidence],
    embedding: List[Evidence],
    *,
    lexical_weight: float,
    embedding_weight: float,
) -> List[Evidence]:
    """
    Fuse lexical and embedding evidence lists into hybrid candidates.

    :param lexical: Lexical evidence list.
    :type lexical: list[Evidence]
    :param embedding: Embedding evidence list.
    :type embedding: list[Evidence]
    :param lexical_weight: Lexical score weight.
    :type lexical_weight: float
    :param embedding_weight: Embedding score weight.
    :type embedding_weight: float
    :return: Hybrid evidence list.
    :rtype: list[Evidence]
    """
    merged: Dict[str, Dict[str, Optional[Evidence]]] = {}
    for evidence_item in lexical:
        merged.setdefault(evidence_item.item_id, {})["lexical"] = evidence_item
    for evidence_item in embedding:
        merged.setdefault(evidence_item.item_id, {})["embedding"] = evidence_item

    candidates: List[Evidence] = []
    for item_id, sources in merged.items():
        lexical_evidence = sources.get("lexical")
        embedding_evidence = sources.get("embedding")
        lexical_score = lexical_evidence.score if lexical_evidence else 0.0
        embedding_score = embedding_evidence.score if embedding_evidence else 0.0
        combined_score = (lexical_score * lexical_weight) + (embedding_score * embedding_weight)
        base_evidence = lexical_evidence or embedding_evidence
        candidates.append(
            Evidence(
                item_id=item_id,
                source_uri=base_evidence.source_uri,
                media_type=base_evidence.media_type,
                score=combined_score,
                rank=1,
                text=base_evidence.text,
                content_ref=base_evidence.content_ref,
                span_start=base_evidence.span_start,
                span_end=base_evidence.span_end,
                stage="hybrid",
                stage_scores={"lexical": lexical_score, "embedding": embedding_score},
                configuration_id="",
                snapshot_id="",
                metadata=base_evidence.metadata,
                hash=base_evidence.hash,
            )
        )
    return candidates
