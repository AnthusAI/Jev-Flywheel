"""
ReinforcementMemory: the main orchestration engine.

Ties together Virtuus persistence, vector store, embedding, clustering,
LLM labeling, and memory weight management into a single coherent API.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ._embedding import EmbedFn
from ._lifecycle import derive_lifecycle
from ._llm import CausalFn, LabelFn, SynthesisFn
from ._models import AnalysisResult, ExemplarRecord, TimestampedText, TopicResult
from ._store import ReinforcementMemoryStore
from ._vector_store import VectorRecord, VectorStore
from ._weights import initial_weight, tier_from_weight, update_memory_weights

logger = logging.getLogger(__name__)

_EXEMPLAR_TEXT_LIMIT = 300


class ReinforcementMemory:
    """
    Living semantic memory for a stream of timestamped texts.

    Maintains topic clusters that are reinforced by new matching texts and
    decay when no new texts arrive.  Persists all state to ``data_dir`` via
    Virtuus and stores cluster centroids in ``vector_store`` for similarity
    search.

    :param data_dir: Root directory for Virtuus JSON files (texts, topics,
        runs).  Created automatically if it does not exist.
    :param vector_store: :class:`~._vector_store.VectorStore` implementation.
        Use :class:`~._vector_store.LocalVectorStore` for local development
        or :class:`~._vector_store.S3VectorStore` for production.
    :param embed: :data:`~._embedding.EmbedFn` callable.  Use
        :func:`~._embedding.hash_embedder` for tests,
        :func:`~._embedding.sentence_transformer_embedder` for production.
    :param label: Optional :data:`~._llm.LabelFn`.  When omitted, labels
        fall back to top keywords.
    :param infer_cause: Optional :data:`~._llm.CausalFn`.  When omitted,
        per-exemplar causal inference is skipped.
    :param synthesize_cause: Optional :data:`~._llm.SynthesisFn`.  When
        omitted, per-topic cause synthesis is skipped.
    :param min_topic_size: Minimum cluster member count passed to
        :class:`~._clusterer.TopicClusterer`.
    :param embedding_dim: Embedding vector dimension.
    :param max_exemplars: Maximum number of exemplars returned per topic.
        Exemplars are selected by proximity to the cluster centroid, then
        sorted most-recent first.  Defaults to 5.
    """

    def __init__(
        self,
        data_dir: str,
        vector_store: VectorStore,
        embed: EmbedFn,
        label: Optional[LabelFn] = None,
        infer_cause: Optional[CausalFn] = None,
        synthesize_cause: Optional[SynthesisFn] = None,
        min_topic_size: int = 10,
        embedding_dim: int = 384,
        max_exemplars: int = 5,
    ) -> None:
        """Initialise the engine and open the Virtuus store."""
        self._store = ReinforcementMemoryStore(data_dir)
        self._vector_store = vector_store
        self._embed = embed
        self._label = label
        self._infer_cause = infer_cause
        self._synthesize_cause = synthesize_cause
        self._min_topic_size = min_topic_size
        self._embedding_dim = embedding_dim
        self._max_exemplars = max_exemplars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, texts: List[TimestampedText]) -> int:
        """
        Persist timestamped texts to the Virtuus store.

        Idempotent: records with the same ``id`` are updated in place.

        :param texts: Text records to store.
        :return: Number of records stored.
        """
        return self._store.put_texts(texts)

    def analyze(
        self,
        group_id: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        min_topic_size: Optional[int] = None,
        cluster_selection_method: str = "leaf",
        cluster_selection_epsilon: float = 0.5,
    ) -> AnalysisResult:
        """
        Run the full pipeline for a group and return the analysis result.

        Steps:

        1. Load texts from Virtuus (optionally filtered by time range).
        2. Embed all texts.
        3. Cluster embeddings via BERTopic / KMeans fallback.
        4. Compute keywords and representative exemplars per cluster.
        5. Generate topic labels (LLM or keyword fallback).
        6. Derive lifecycle tiers from member timestamps.
        7. Reinforce / decay weights against prior topics.
        8. Optionally infer and synthesize root causes.
        9. Persist cluster state to vector store and Virtuus.
        10. Return :class:`~._models.AnalysisResult`.

        :param group_id: Group to analyze.
        :param since: Optional ISO timestamp lower bound for text retrieval.
        :param until: Optional ISO timestamp upper bound for text retrieval.
        :param min_topic_size: Override the instance ``min_topic_size``.
        :param cluster_selection_method: BERTopic HDBSCAN method.
        :param cluster_selection_epsilon: BERTopic HDBSCAN epsilon.
        :return: Analysis result with topics, weights, and lifecycle tiers.
        """
        from ._clusterer import TopicClusterer

        now = datetime.now(timezone.utc)
        run_id = str(uuid.uuid4())

        # 1. Load texts
        raw = self._store.get_texts_for_group(group_id, since=since, until=until)
        if not raw:
            return AnalysisResult(
                group_id=group_id,
                topics=[],
                texts_analyzed=0,
                cluster_version="",
                run_id=run_id,
            )

        texts = [r["text"] for r in raw]
        timestamps = [r.get("timestamp", "") for r in raw]
        text_ids = [r["id"] for r in raw]
        metadatas = [r.get("metadata", {}) for r in raw]

        # 2. Embed
        embeddings = self._embed(texts)

        # 3. Cluster
        mt = min_topic_size or self._min_topic_size

        clusterer = TopicClusterer(
            min_topic_size=min(mt, max(2, len(texts) // 3)),
            label_generator=None,  # Labels generated separately below
        )
        topic_ids, cluster_version = clusterer.cluster(
            embeddings,
            texts,
            min_topic_size=min(mt, max(2, len(texts) // 3)),
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

        # 4 + 5. Build per-topic data
        prior_topics = self._store.get_topics_for_group(group_id)

        raw_topics: List[Dict[str, Any]] = []
        for tid in sorted(set(topic_ids)):
            if tid == -1:
                continue
            member_mask = topic_ids == tid
            member_indices = list(np.where(member_mask)[0])
            member_count = int(np.sum(member_mask))

            keywords = clusterer.get_keywords(int(tid), n=8)

            # Select up to max_exemplars by centroid proximity, then sort
            # most-recent first so callers can show the freshest examples.
            exemplar_pairs = clusterer.get_representative_exemplars(
                int(tid), n=self._max_exemplars
            )
            exemplars = []
            causal_contexts: List[Dict[str, Any]] = []
            for ex_idx, ex_text in exemplar_pairs:
                truncated = (
                    ex_text[:_EXEMPLAR_TEXT_LIMIT] + "…"
                    if len(ex_text) > _EXEMPLAR_TEXT_LIMIT
                    else ex_text
                )
                exemplars.append(
                    ExemplarRecord(
                        text=truncated,
                        text_id=text_ids[ex_idx],
                        metadata=metadatas[ex_idx],
                        timestamp=timestamps[ex_idx] if ex_idx < len(timestamps) else None,
                    )
                )
                causal_contexts.append({"edit_comment": ex_text})

            # Sort exemplars most-recent first (ISO 8601 strings compare lexicographically).
            exemplars.sort(key=lambda e: e.timestamp or "", reverse=True)

            # 5. Labels
            label_str: str
            if self._label and keywords:
                try:
                    label_str = self._label(keywords, [e.text for e in exemplars])
                except Exception as exc:
                    logger.warning("Label generation failed for topic %s: %s", tid, exc)
                    label_str = ", ".join(keywords[:3]) if keywords else f"Topic {tid}"
            else:
                label_str = ", ".join(keywords[:3]) if keywords else f"Topic {tid}"

            # 6. Lifecycle
            member_timestamps = [timestamps[i] for i in member_indices if i < len(timestamps)]
            lifecycle_tier, is_new, is_trending, days_inactive = derive_lifecycle(
                member_timestamps, now=now
            )

            raw_topics.append(
                {
                    "tid": int(tid),
                    "label": label_str,
                    "keywords": keywords,
                    "exemplars": exemplars,
                    "member_count": member_count,
                    "lifecycle_tier": lifecycle_tier,
                    "is_new": is_new,
                    "is_trending": is_trending,
                    "days_inactive": days_inactive,
                    "member_indices": member_indices,
                    "_causal_contexts": causal_contexts,
                    "_member_timestamps": member_timestamps,
                }
            )

        # 7. Reinforce / decay weights against prior topics
        prior_cluster_dicts = []
        for pt in prior_topics:
            cid = pt.get("topic_id", "")
            # Extract numeric id from "group_id::N" format
            numeric = _extract_numeric_id(cid)
            prior_cluster_dicts.append(
                {
                    "cluster_id": numeric,
                    "memory_weight": pt.get("memory_weight", initial_weight()),
                    "memory_tier": pt.get("memory_tier", "warm"),
                }
            )

        active_ids = [t["tid"] for t in raw_topics]
        days_inactive_map = {t["tid"]: t["days_inactive"] or 0 for t in raw_topics}
        updated_priors, _ = update_memory_weights(
            prior_cluster_dicts,
            active_cluster_ids=active_ids,
            days_inactive=days_inactive_map,
            prune=True,
        )
        prior_weights = {c["cluster_id"]: c["memory_weight"] for c in updated_priors}

        # 8. Causal inference
        causal_by_tid: Dict[int, List[str]] = {}
        if self._infer_cause:
            for t in raw_topics:
                causes = []
                for ctx in t["_causal_contexts"]:
                    try:
                        cause = self._infer_cause(ctx["edit_comment"], ctx)
                        if cause:
                            causes.append(cause)
                    except Exception as exc:
                        logger.warning("Causal inference failed for topic %s: %s", t["tid"], exc)
                causal_by_tid[t["tid"]] = causes

        root_causes: Dict[int, Optional[str]] = {}
        if self._synthesize_cause and causal_by_tid:
            for t in raw_topics:
                causes = causal_by_tid.get(t["tid"], [])
                if causes:
                    try:
                        root_causes[t["tid"]] = self._synthesize_cause(
                            t["label"], t["keywords"], causes
                        )
                    except Exception as exc:
                        logger.warning("Cause synthesis failed for topic %s: %s", t["tid"], exc)

        # 9. Assemble results + persist
        topic_results: List[TopicResult] = []
        now_iso = now.isoformat()

        # Clear previous topic state for this group
        self._store.delete_topics_for_group(group_id)

        # Clear and rebuild vector store for this group's clusters
        self._vector_store.delete_all()

        for t in raw_topics:
            tid = t["tid"]
            weight = prior_weights.get(tid, initial_weight())
            tier = tier_from_weight(weight)

            topic_id_str = f"{group_id}::{tid}"
            root_cause = root_causes.get(tid)

            # Persist topic to Virtuus
            self._store.put_topic(
                {
                    "topic_id": topic_id_str,
                    "group_id": group_id,
                    "label": t["label"],
                    "keywords": t["keywords"],
                    "member_count": t["member_count"],
                    "memory_weight": weight,
                    "memory_tier": tier,
                    "lifecycle_tier": t["lifecycle_tier"],
                    "is_new": t["is_new"],
                    "is_trending": t["is_trending"],
                    "days_inactive": t["days_inactive"],
                    "root_cause": root_cause,
                    "cluster_version": cluster_version,
                    "last_updated": now_iso,
                }
            )

            # Build centroid vector record
            centroid = clusterer.cluster_centroids().get(tid)
            if centroid is not None:
                self._vector_store.put_vectors(
                    [
                        VectorRecord(
                            key=f"cluster:{group_id}::{tid}",
                            embedding=centroid.tolist(),
                            metadata={
                                "group_id": group_id,
                                "topic_id": topic_id_str,
                                "label": t["label"],
                                "member_count": t["member_count"],
                                "memory_weight": weight,
                                "memory_tier": tier,
                            },
                        )
                    ]
                )

            topic_results.append(
                TopicResult(
                    topic_id=tid,
                    label=t["label"],
                    keywords=t["keywords"],
                    exemplars=t["exemplars"],
                    member_count=t["member_count"],
                    memory_weight=weight,
                    memory_tier=tier,
                    lifecycle_tier=t["lifecycle_tier"],
                    is_new=t["is_new"],
                    is_trending=t["is_trending"],
                    days_inactive=t["days_inactive"],
                    root_cause=root_cause,
                )
            )

        # Record the run
        self._store.record_run(
            {
                "run_id": run_id,
                "group_id": group_id,
                "timestamp": now_iso,
                "texts_analyzed": len(texts),
                "topics_found": len(topic_results),
                "cluster_version": cluster_version,
            }
        )

        return AnalysisResult(
            group_id=group_id,
            topics=topic_results,
            texts_analyzed=len(texts),
            cluster_version=cluster_version,
            run_id=run_id,
        )

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
    ):
        """
        Find the nearest topic cluster centroids for a query embedding.

        :param embedding: Query vector.
        :param k: Maximum number of results.
        :param threshold: Optional minimum similarity score.
        :return: List of :class:`~._models.QueryResult` objects.
        """
        return self._vector_store.query_nearest(embedding, k=k, threshold=threshold)

    def get_topics(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve persisted topic state from Virtuus.

        :param group_id: Group identifier.
        :return: List of topic dicts from the store.
        """
        return self._store.get_topics_for_group(group_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_numeric_id(topic_id_str: str) -> int:
    """Extract the trailing integer from a ``group_id::N`` topic ID."""
    if "::" in topic_id_str:
        suffix = topic_id_str.split("::")[-1]
        if suffix.lstrip("-").isdigit():
            return int(suffix)
    if isinstance(topic_id_str, str) and topic_id_str.lstrip("-").isdigit():
        return int(topic_id_str)
    return hash(topic_id_str) % (2**31)
