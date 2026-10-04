"""
Data models for Reinforcement Memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimestampedText:
    """
    A single text with a timestamp and optional metadata.

    :ivar id: Stable unique identifier for this text record.
    :vartype id: str
    :ivar group_id: Partitions texts into independent analysis groups.
    :vartype group_id: str
    :ivar timestamp: ISO 8601 timestamp (e.g. ``2024-01-15T10:00:00Z``).
    :vartype timestamp: str
    :ivar text: The text content to analyze.
    :vartype text: str
    :ivar metadata: Arbitrary key-value metadata preserved alongside the text.
    :vartype metadata: dict[str, Any]
    """

    id: str
    group_id: str
    timestamp: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExemplarRecord:
    """
    A representative example from a topic cluster.

    :ivar text: Truncated text of the exemplar.
    :vartype text: str
    :ivar text_id: ID of the source :class:`TimestampedText`.
    :vartype text_id: str
    :ivar metadata: Metadata from the source text.
    :vartype metadata: dict[str, Any]
    :ivar timestamp: ISO 8601 timestamp of the source text, for recency ordering.
    :vartype timestamp: str | None
    """

    text: str
    text_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class TopicResult:
    """
    A discovered topic cluster with memory state.

    :ivar topic_id: Integer cluster identifier.
    :vartype topic_id: int
    :ivar label: Human-readable topic label (LLM-generated or keyword-based).
    :vartype label: str
    :ivar keywords: Top keywords characterizing the cluster.
    :vartype keywords: list[str]
    :ivar exemplars: Representative text samples closest to the centroid.
    :vartype exemplars: list[ExemplarRecord]
    :ivar member_count: Number of texts in this cluster.
    :vartype member_count: int
    :ivar memory_weight: Reinforcement weight in [0.0, 1.0].
    :vartype memory_weight: float
    :ivar memory_tier: ``hot``, ``warm``, or ``cold`` based on weight.
    :vartype memory_tier: str
    :ivar lifecycle_tier: ``new``, ``trending``, or ``established`` based on timestamps.
    :vartype lifecycle_tier: str
    :ivar is_new: True when cluster has only short-term members.
    :vartype is_new: bool
    :ivar is_trending: True when cluster has recent members but no long-term ones.
    :vartype is_trending: bool
    :ivar days_inactive: Days since the most recent member timestamp, or None.
    :vartype days_inactive: int | None
    :ivar root_cause: LLM-inferred root cause statement, or None.
    :vartype root_cause: str | None
    """

    topic_id: int
    label: str
    keywords: List[str]
    exemplars: List[ExemplarRecord]
    member_count: int
    memory_weight: float
    memory_tier: str
    lifecycle_tier: str
    is_new: bool
    is_trending: bool
    days_inactive: Optional[int]
    root_cause: Optional[str]


@dataclass
class AnalysisResult:
    """
    Result of a single :meth:`ReinforcementMemory.analyze` run.

    :ivar group_id: Group identifier this analysis covers.
    :vartype group_id: str
    :ivar topics: Discovered topic clusters, ordered by topic_id.
    :vartype topics: list[TopicResult]
    :ivar texts_analyzed: Number of texts included in this analysis.
    :vartype texts_analyzed: int
    :ivar cluster_version: Timestamp string identifying this clustering run.
    :vartype cluster_version: str
    :ivar run_id: Unique identifier for this analysis run.
    :vartype run_id: str
    """

    group_id: str
    topics: List[TopicResult]
    texts_analyzed: int
    cluster_version: str
    run_id: str


@dataclass
class VectorRecord:
    """
    A single vector with metadata for storage in a :class:`VectorStore`.

    :ivar key: Unique key identifying this vector.
    :vartype key: str
    :ivar embedding: Float32 embedding vector.
    :vartype embedding: list[float]
    :ivar metadata: Arbitrary metadata stored alongside the vector.
    :vartype metadata: dict[str, Any]
    """

    key: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """
    A single result from a vector similarity query.

    :ivar key: Key of the matched vector.
    :vartype key: str
    :ivar distance: Cosine distance (lower = more similar).
    :vartype distance: float
    :ivar similarity: Cosine similarity = 1 - distance.
    :vartype similarity: float
    :ivar metadata: Metadata stored with the matched vector.
    :vartype metadata: dict[str, Any]
    """

    key: str
    distance: float
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
