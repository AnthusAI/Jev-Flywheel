"""
TopicClusterer: clusters pre-computed embeddings via BERTopic (UMAP + HDBSCAN).

Decouples embedding from clustering. Computes centroids, p95 distance
boundaries, TF-IDF keywords, and representative exemplars. Requires the
``topic-modeling`` optional dependency group.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 384


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance (1 - cosine similarity) between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    sim = np.dot(a, b) / (a_norm * b_norm)
    return float(1.0 - np.clip(sim, -1.0, 1.0))


class TopicClusterer:
    """
    Cluster pre-computed embeddings via BERTopic (UMAP + HDBSCAN).

    Computes cluster centroids, p95 distance boundaries, TF-IDF keywords, and
    representative exemplars.  Falls back to KMeans for small datasets.

    Requires ``bertopic``, ``umap-learn``, and ``hdbscan`` (available via the
    ``topic-modeling`` optional dependency group).

    :param min_topic_size: Minimum cluster member count for HDBSCAN.
    :param umap_n_components: Target dimensionality for UMAP reduction.
    :param umap_min_dist: UMAP ``min_dist`` parameter.
    :param umap_metric: Distance metric for UMAP.
    :param label_generator: Optional callable ``(exemplar_texts) -> label``.
        Called by :meth:`generate_labels` when provided.
    """

    def __init__(
        self,
        min_topic_size: int = 10,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        label_generator: Optional[Callable[[List[str]], str]] = None,
    ) -> None:
        """Initialise the clusterer with configuration."""
        self.min_topic_size = min_topic_size
        self.umap_n_components = umap_n_components
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self._label_generator = label_generator
        self._topics: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None
        self._documents: Optional[List[str]] = None
        self._cluster_version: Optional[str] = None
        self._topic_model = None

    def cluster(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        min_topic_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        cluster_selection_method: str = "leaf",
        cluster_selection_epsilon: float = 0.5,
    ) -> Tuple[np.ndarray, str]:
        """
        Cluster embeddings and return ``(topic_ids, cluster_version)``.

        ``topic_ids[i]`` is the cluster assignment for ``documents[i]``.
        Outliers are assigned ``-1``.

        For datasets smaller than 15 items, KMeans is used instead of
        BERTopic to avoid numerical instability in UMAP.  If HDBSCAN assigns
        every point as an outlier, KMeans is used as a fallback.

        :param embeddings: Pre-computed embedding matrix, shape ``(n, dim)``.
        :param documents: Text strings corresponding to each embedding row.
        :param min_topic_size: Override the instance ``min_topic_size``.
        :param min_samples: HDBSCAN ``min_samples`` parameter.
        :param cluster_selection_method: HDBSCAN ``cluster_selection_method``.
        :param cluster_selection_epsilon: HDBSCAN ``cluster_selection_epsilon``.
        :return: ``(topic_ids, cluster_version)`` tuple.
        """
        n = len(embeddings)
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        if n < 15:
            topics = self._kmeans_fallback(embeddings, n)
            self._store(np.array(topics), embeddings, documents, version)
            return self._topics, self._cluster_version  # type: ignore[return-value]

        mt = min(min_topic_size or self.min_topic_size, max(2, n))
        ms = min_samples if min_samples is not None else min(2, mt)
        n_neighbors = min(15, max(2, n - 1))
        n_components = min(self.umap_n_components, max(2, n - 2))
        init_method = "spectral"

        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from umap import UMAP
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Reinforcement Memory topic clustering requires BERTopic, "
                "HDBSCAN, and UMAP. Install them with: "
                'pip install "biblicus[reinforcement-memory]"'
            ) from exc

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            init=init_method,
            random_state=42,
            n_jobs=1,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=mt,
            min_samples=ms,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric="euclidean",
            prediction_data=True,
        )
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(documents, embeddings)
        topics = np.array(topics)

        if np.all(topics == -1) and n >= mt:
            logger.warning(
                "HDBSCAN returned all outliers; falling back to KMeans on raw embeddings"
            )
            topics = self._kmeans_fallback(embeddings, n)
            self._topic_model = None
        else:
            self._topic_model = topic_model

        self._store(np.array(topics), embeddings, documents, version)
        return self._topics, self._cluster_version  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Cluster statistics
    # ------------------------------------------------------------------

    def cluster_centroids(self) -> Dict[int, np.ndarray]:
        """
        Return the centroid (mean embedding) for each non-outlier cluster.

        :return: Mapping of cluster_id to centroid vector.
        """
        if self._topics is None or self._embeddings is None:
            return {}
        centroids: Dict[int, np.ndarray] = {}
        for tid in np.unique(self._topics):
            if tid == -1:
                continue
            mask = self._topics == tid
            members = self._embeddings[mask]
            centroids[int(tid)] = np.mean(members, axis=0).astype(np.float32)
        return centroids

    def cluster_boundaries(self) -> Dict[int, float]:
        """
        Return the p95 cosine distance from members to centroid per cluster.

        :return: Mapping of cluster_id to p95 boundary distance.
        """
        centroids = self.cluster_centroids()
        if not centroids:
            return {}
        boundaries: Dict[int, float] = {}
        for tid, centroid in centroids.items():
            mask = self._topics == tid
            members = self._embeddings[mask]  # type: ignore[index]
            distances = [_cosine_distance(m, centroid) for m in members]
            boundaries[tid] = float(np.percentile(distances, 95))
        return boundaries

    def get_keywords(self, topic_id: int, n: int = 8) -> List[str]:
        """
        Extract top TF-IDF keywords for a cluster.

        :param topic_id: Cluster identifier.
        :param n: Maximum number of keywords to return.
        :return: List of keyword strings.
        """
        if self._documents is None:
            return []
        mask = self._topics == topic_id
        indices = np.where(mask)[0]
        if len(indices) < 2:
            return []
        docs = [self._documents[i] for i in indices]
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=1,
            )
            X = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            scores = np.asarray(X.sum(axis=0)).flatten()
            top_indices = scores.argsort()[-n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0 and feature_names[i]]
        except Exception as exc:
            logger.warning("Failed to extract keywords for topic %s: %s", topic_id, exc)
            return []

    def get_representative_exemplars(self, topic_id: int, n: int = 5) -> List[Tuple[int, str]]:
        """
        Return exemplar ``(original_index, text)`` pairs nearest the centroid.

        The ``original_index`` is the position in the list passed to
        :meth:`cluster`, enabling callers to map back to document IDs or
        metadata.

        :param topic_id: Cluster identifier.
        :param n: Maximum number of exemplars to return.
        :return: List of ``(index, text)`` tuples ordered by proximity to centroid.
        """
        if self._documents is None or self._embeddings is None:
            return []
        mask = self._topics == topic_id
        indices = np.where(mask)[0]
        centroid = self.cluster_centroids().get(topic_id)
        if centroid is None:
            return []
        distances = [(_cosine_distance(self._embeddings[i], centroid), int(i)) for i in indices]
        distances.sort(key=lambda x: x[0])
        return [(idx, self._documents[idx]) for _, idx in distances[:n]]

    def generate_labels(self) -> Dict[int, str]:
        """
        Generate a label for each cluster.

        Uses ``label_generator`` when provided; otherwise returns
        ``"Topic {id}"``.

        :return: Mapping of cluster_id to label string.
        """
        centroids = self.cluster_centroids()
        labels: Dict[int, str] = {}
        for tid in centroids:
            docs = [doc for _, doc in self.get_representative_exemplars(tid)]
            if self._label_generator:
                labels[tid] = self._label_generator(docs)
            else:
                labels[tid] = f"Topic {tid}"
        return labels

    def get_cluster_records(self) -> List[Dict[str, Any]]:
        """
        Build cluster summary records suitable for vector store persistence.

        :return: List of dicts with centroid, label, keywords, and member stats.
        """
        centroids = self.cluster_centroids()
        boundaries = self.cluster_boundaries()
        labels = self.generate_labels()
        records: List[Dict[str, Any]] = []
        for tid in centroids:
            mask = self._topics == tid
            member_count = int(np.sum(mask))
            label = labels.get(tid, f"Topic {tid}")
            p95 = boundaries.get(tid, 0.0)
            centroid_list = centroids[tid].tolist()
            records.append(
                {
                    "doc_id": f"cluster-{tid}",
                    "text": label,
                    "embedding": centroid_list,
                    "metadata": {
                        "p95_distance": p95,
                        "label": label,
                        "member_count": member_count,
                    },
                    "cluster_id": str(tid),
                    "cluster_version": self._cluster_version or "",
                    "record_type": "cluster",
                    "centroid_embedding": centroid_list,
                    "p95_distance": p95,
                    "label": label,
                    "member_count": member_count,
                }
            )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kmeans_fallback(self, embeddings: np.ndarray, n: int) -> np.ndarray:
        """Run KMeans and return topic_ids array."""
        from sklearn.cluster import KMeans

        k = max(1, min(3, n // 3))
        if k <= 1 or n < 2:
            return np.zeros(n, dtype=int)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings)

    def _store(
        self,
        topics: np.ndarray,
        embeddings: np.ndarray,
        documents: List[str],
        version: str,
    ) -> None:
        """Persist clustering results on the instance."""
        self._topics = topics
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        self._documents = documents
        self._cluster_version = version
