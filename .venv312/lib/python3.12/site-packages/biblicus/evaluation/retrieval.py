"""
Evaluation utilities for Biblicus retrieval snapshots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import DATASET_SCHEMA_VERSION
from ..corpus import Corpus
from ..models import QueryBudget, RetrievalResult, RetrievalSnapshot
from ..retrievers import get_retriever
from ..time import utc_now_iso


class EvaluationQuery(BaseModel):
    """
    Query record for retrieval evaluation.

    :ivar query_id: Unique identifier for the query.
    :vartype query_id: str
    :ivar query_text: Natural language query to execute.
    :vartype query_text: str
    :ivar expected_item_id: Optional expected item identifier.
    :vartype expected_item_id: str or None
    :ivar expected_source_uri: Optional expected source uniform resource identifier.
    :vartype expected_source_uri: str or None
    :ivar kind: Query kind (gold or synthetic).
    :vartype kind: str
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query_text: str
    expected_item_id: Optional[str] = None
    expected_source_uri: Optional[str] = None
    kind: str = Field(default="gold")

    @model_validator(mode="after")
    def _require_expectation(self) -> "EvaluationQuery":
        if not self.expected_item_id and not self.expected_source_uri:
            raise ValueError(
                "Evaluation queries must include expected_item_id or expected_source_uri"
            )
        return self


class EvaluationDataset(BaseModel):
    """
    Dataset for retrieval evaluation.

    :ivar schema_version: Dataset schema version.
    :vartype schema_version: int
    :ivar name: Dataset name.
    :vartype name: str
    :ivar description: Optional description.
    :vartype description: str or None
    :ivar queries: List of evaluation queries.
    :vartype queries: list[EvaluationQuery]
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    name: str
    description: Optional[str] = None
    queries: List[EvaluationQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_schema_version(self) -> "EvaluationDataset":
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema version: {self.schema_version}")
        return self


class EvaluationResult(BaseModel):
    """
    Result bundle for a retrieval evaluation.

    :ivar dataset: Dataset metadata.
    :vartype dataset: dict[str, object]
    :ivar retriever_id: Retriever identifier.
    :vartype retriever_id: str
    :ivar snapshot_id: Retrieval snapshot identifier.
    :vartype snapshot_id: str
    :ivar evaluated_at: International Organization for Standardization 8601 evaluation timestamp.
    :vartype evaluated_at: str
    :ivar metrics: Quality metrics for retrieval.
    :vartype metrics: dict[str, float]
    :ivar system: System metrics for retrieval.
    :vartype system: dict[str, float]
    """

    model_config = ConfigDict(extra="forbid")

    dataset: Dict[str, object]
    retriever_id: str
    snapshot_id: str
    evaluated_at: str
    metrics: Dict[str, float]
    system: Dict[str, float]


def load_dataset(path: Path) -> EvaluationDataset:
    """
    Load an evaluation dataset from JavaScript Object Notation.

    :param path: Path to the dataset JavaScript Object Notation file.
    :type path: Path
    :return: Parsed evaluation dataset.
    :rtype: EvaluationDataset
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(data)


def evaluate_snapshot(
    *,
    corpus: Corpus,
    snapshot: RetrievalSnapshot,
    dataset: EvaluationDataset,
    budget: QueryBudget,
) -> EvaluationResult:
    """
    Evaluate a retrieval snapshot against a dataset.

    :param corpus: Corpus associated with the snapshot.
    :type corpus: Corpus
    :param snapshot: Retrieval snapshot manifest.
    :type snapshot: RetrievalSnapshot
    :param dataset: Evaluation dataset.
    :type dataset: EvaluationDataset
    :param budget: Evidence selection budget.
    :type budget: QueryBudget
    :return: Evaluation result bundle.
    :rtype: EvaluationResult
    """
    retriever = get_retriever(snapshot.configuration.retriever_id)
    latency_seconds: List[float] = []
    hit_count = 0
    reciprocal_ranks: List[float] = []

    for query in dataset.queries:
        timer_start = time.perf_counter()
        result = retriever.query(
            corpus, snapshot=snapshot, query_text=query.query_text, budget=budget
        )
        elapsed_seconds = time.perf_counter() - timer_start
        latency_seconds.append(elapsed_seconds)
        expected_rank = _expected_rank(result, query)
        if expected_rank is not None:
            hit_count += 1
            reciprocal_ranks.append(1.0 / expected_rank)
        else:
            reciprocal_ranks.append(0.0)

    total_queries = max(len(dataset.queries), 1)
    max_total_items = float(budget.max_total_items)
    hit_rate = hit_count / total_queries
    precision_at_max_total_items = hit_count / (total_queries * max_total_items)
    mean_reciprocal_rank = sum(reciprocal_ranks) / total_queries

    metrics = {
        "hit_rate": hit_rate,
        "precision_at_max_total_items": precision_at_max_total_items,
        "mean_reciprocal_rank": mean_reciprocal_rank,
    }
    system = {
        "average_latency_milliseconds": _average_latency_milliseconds(latency_seconds),
        "percentile_95_latency_milliseconds": _percentile_95_latency_milliseconds(latency_seconds),
        "index_bytes": float(_snapshot_artifact_bytes(corpus, snapshot)),
    }
    dataset_meta = {
        "name": dataset.name,
        "description": dataset.description,
        "queries": len(dataset.queries),
    }
    return EvaluationResult(
        dataset=dataset_meta,
        retriever_id=snapshot.configuration.retriever_id,
        snapshot_id=snapshot.snapshot_id,
        evaluated_at=utc_now_iso(),
        metrics=metrics,
        system=system,
    )


def _expected_rank(result: RetrievalResult, query: EvaluationQuery) -> Optional[int]:
    """
    Locate the first evidence rank that matches the expected item or source.

    :param result: Retrieval result for a query.
    :type result: RetrievalResult
    :param query: Evaluation query definition.
    :type query: EvaluationQuery
    :return: Rank of the first matching evidence item, or None.
    :rtype: int or None
    """
    for evidence in result.evidence:
        if query.expected_item_id and evidence.item_id == query.expected_item_id:
            return evidence.rank
        if query.expected_source_uri and evidence.source_uri == query.expected_source_uri:
            return evidence.rank
    return None


def _average_latency_milliseconds(latencies: List[float]) -> float:
    """
    Compute average latency in milliseconds.

    :param latencies: Latency samples in seconds.
    :type latencies: list[float]
    :return: Average latency in milliseconds.
    :rtype: float
    """
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies) * 1000.0


def _percentile_95_latency_milliseconds(latencies: List[float]) -> float:
    """
    Compute the percentile 95 latency in milliseconds.

    :param latencies: Latency samples in seconds.
    :type latencies: list[float]
    :return: Percentile 95 latency in milliseconds.
    :rtype: float
    """
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    percentile_index = int(round(0.95 * (len(sorted_latencies) - 1)))
    return sorted_latencies[percentile_index] * 1000.0


def _snapshot_artifact_bytes(corpus: Corpus, snapshot: RetrievalSnapshot) -> int:
    """
    Sum artifact sizes for a retrieval snapshot.

    :param corpus: Corpus that owns the artifacts.
    :type corpus: Corpus
    :param snapshot: Retrieval snapshot manifest.
    :type snapshot: RetrievalSnapshot
    :return: Total artifact bytes.
    :rtype: int
    """
    total_bytes = 0
    for artifact_relpath in snapshot.snapshot_artifacts:
        artifact_path = corpus.root / artifact_relpath
        if artifact_path.exists():
            total_bytes += artifact_path.stat().st_size
    return total_bytes
