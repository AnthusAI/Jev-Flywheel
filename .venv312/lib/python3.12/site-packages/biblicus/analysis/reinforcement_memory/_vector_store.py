"""
Vector store abstraction for Reinforcement Memory.

Provides a :class:`VectorStore` protocol and two concrete implementations:

- :class:`LocalVectorStore` — file-backed store using JSON files and
  brute-force cosine similarity.  No external dependencies beyond NumPy.
  Suitable for development and testing.

- :class:`S3VectorStore` — AWS S3 Vectors-backed store using boto3.
  Requires the ``aws`` optional dependency group.  Suitable for production.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

import numpy as np

from ._models import QueryResult, VectorRecord

logger = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 500
_DEFAULT_EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol for a vector similarity store.

    Implementations must support storing, listing, querying, and deleting
    vectors.  All vectors share the same embedding dimension, configured at
    construction time.
    """

    def health_check(self) -> bool:
        """Return True if the store is reachable and operational."""
        pass

    def put_vectors(self, vectors: List[VectorRecord]) -> None:
        """
        Upsert a list of vectors.

        :param vectors: Vector records to store.
        """
        pass

    def query_nearest(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[QueryResult]:
        """
        Find the ``k`` nearest vectors to ``embedding``.

        :param embedding: Query vector.
        :param k: Maximum number of results.
        :param threshold: Optional minimum similarity score for results.
        :return: Results ordered by descending similarity.
        """
        pass

    def delete_all(self) -> int:
        """
        Delete all vectors from the store.

        :return: Number of vectors deleted.
        """
        pass

    def list_all(self) -> List[VectorRecord]:
        """
        Return all stored vectors.

        :return: All :class:`~._models.VectorRecord` objects in the store.
        """
        pass


# ---------------------------------------------------------------------------
# LocalVectorStore
# ---------------------------------------------------------------------------


class LocalVectorStore:
    """
    File-backed vector store for local / offline use.

    Each vector is stored as a JSON file named ``{key}.json`` under
    ``store_dir``.  Similarity search uses brute-force cosine distance.

    :param store_dir: Directory for JSON vector files.
    :param embedding_dim: Expected embedding dimension.
    """

    def __init__(
        self,
        store_dir: str,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Initialise the local store."""
        self._dir = store_dir
        self._dim = embedding_dim
        os.makedirs(store_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True — local store is always available."""
        return True

    def put_vectors(self, vectors: List[VectorRecord]) -> None:
        """
        Write vector records to disk.

        :param vectors: Records to store.
        """
        for vec in vectors:
            path = self._path(vec.key)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "key": vec.key,
                        "embedding": [float(v) for v in vec.embedding],
                        "metadata": vec.metadata,
                    },
                    fh,
                )

    def query_nearest(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[QueryResult]:
        """
        Brute-force cosine similarity search over all stored vectors.

        :param embedding: Query embedding.
        :param k: Number of nearest results.
        :param threshold: Optional minimum similarity cutoff.
        :return: Results ordered by descending similarity.
        """
        query = np.asarray(embedding, dtype=np.float32)
        results: List[QueryResult] = []
        for rec in self.list_all():
            vec = np.asarray(rec.embedding, dtype=np.float32)
            sim = _cosine_similarity(query, vec)
            if threshold is not None and sim < threshold:
                continue
            results.append(
                QueryResult(
                    key=rec.key,
                    distance=float(1.0 - sim),
                    similarity=float(sim),
                    metadata=rec.metadata,
                )
            )
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:k]

    def delete_all(self) -> int:
        """
        Remove all JSON vector files.

        :return: Number of files deleted.
        """
        count = 0
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(self._dir, fname))
                count += 1
        return count

    def list_all(self) -> List[VectorRecord]:
        """
        Load and return all stored vectors.

        :return: All :class:`~._models.VectorRecord` objects.
        """
        records: List[VectorRecord] = []
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self._dir, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                records.append(
                    VectorRecord(
                        key=data["key"],
                        embedding=data["embedding"],
                        metadata=data.get("metadata", {}),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed vector file %s: %s", fname, exc)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(":", "_")
        return os.path.join(self._dir, f"{safe}.json")


# ---------------------------------------------------------------------------
# S3VectorStore
# ---------------------------------------------------------------------------


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _to_float_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(v) for v in vector]


class S3VectorStore:
    """
    AWS S3 Vectors-backed store.

    Requires ``boto3`` (available via the ``aws`` optional dependency group).

    :param bucket_name: S3 Vectors bucket name.
    :param index_name: S3 Vectors index name.
    :param region: AWS region.
    :param index_arn: Optional index ARN.  When provided, the ARN is used
        for API calls instead of ``bucket_name + index_name``.
    :param embedding_dim: Expected embedding dimension (default 384).
    :param client: Optional pre-built boto3 s3vectors client (for testing).
    """

    def __init__(
        self,
        bucket_name: str,
        index_name: str,
        region: str,
        index_arn: Optional[str] = None,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
        client=None,
    ) -> None:
        """Initialise the S3 vector store."""
        self.bucket_name = bucket_name
        self.index_name = index_name
        self.index_arn = index_arn
        self.region = region
        self._dim = embedding_dim
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("s3vectors", region_name=region)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ref(self) -> Dict[str, Any]:
        if self.index_arn:
            return {"indexArn": self.index_arn}
        return {"vectorBucketName": self.bucket_name, "indexName": self.index_name}

    def _list_all_keys(self) -> List[str]:
        keys: List[str] = []
        next_token = None
        while True:
            params = {
                **self._ref(),
                "maxResults": _MAX_BATCH_SIZE,
                "returnData": False,
                "returnMetadata": False,
            }
            if next_token:
                params["nextToken"] = next_token
            resp = self._client.list_vectors(**params)
            for v in resp.get("vectors", []):
                if v.get("key"):
                    keys.append(v["key"])
            next_token = resp.get("nextToken")
            if not next_token:
                break
        return keys

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Return True if the S3 Vectors index is reachable.

        :return: True on success, False on any error.
        """
        try:
            self._client.get_index(**self._ref())
            return True
        except Exception as exc:
            logger.error("S3VectorStore health check failed: %s", exc)
            return False

    def put_vectors(self, vectors: List[VectorRecord]) -> None:
        """
        Upsert vectors in batches of up to 500.

        :param vectors: Records to store.
        """
        items = []
        for rec in vectors:
            emb = _to_float_list(rec.embedding)
            if len(emb) != self._dim:
                logger.warning(
                    "Skipping vector %s: expected %d dims, got %d",
                    rec.key,
                    self._dim,
                    len(emb),
                )
                continue
            items.append(
                {
                    "key": rec.key,
                    "data": {"float32": emb},
                    "metadata": rec.metadata,
                }
            )
        for batch in _chunked(items, _MAX_BATCH_SIZE):
            self._client.put_vectors(**self._ref(), vectors=batch)

    def query_nearest(
        self,
        embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[QueryResult]:
        """
        Query the S3 Vectors index for nearest neighbours.

        :param embedding: Query vector.
        :param k: Number of candidates to fetch.
        :param threshold: Optional minimum similarity score.
        :return: Results ordered by descending similarity.
        """
        emb_list = _to_float_list(embedding)
        if len(emb_list) != self._dim:
            return []
        try:
            resp = self._client.query_vectors(
                **self._ref(),
                topK=k,
                queryVector={"float32": emb_list},
                returnMetadata=True,
                returnDistance=True,
            )
        except Exception as exc:
            logger.error("S3VectorStore query failed: %s", exc)
            return []

        results: List[QueryResult] = []
        for hit in resp.get("vectors", []):
            distance = hit.get("distance")
            similarity = None if distance is None else float(1.0 - float(distance))
            if threshold is not None and (similarity is None or similarity < threshold):
                continue
            results.append(
                QueryResult(
                    key=hit.get("key", ""),
                    distance=float(distance) if distance is not None else 1.0,
                    similarity=similarity if similarity is not None else 0.0,
                    metadata=dict(hit.get("metadata", {}) or {}),
                )
            )
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def delete_all(self) -> int:
        """
        Delete all vectors from the index.

        :return: Number of vectors deleted.
        """
        keys = self._list_all_keys()
        if not keys:
            return 0
        deleted = 0
        for batch in _chunked(keys, _MAX_BATCH_SIZE):
            self._client.delete_vectors(**self._ref(), keys=batch)
            deleted += len(batch)
        return deleted

    def list_all(self) -> List[VectorRecord]:
        """
        Retrieve all vectors from the index.

        :return: All :class:`~._models.VectorRecord` objects.
        """
        records: List[VectorRecord] = []
        next_token = None
        while True:
            params = {
                **self._ref(),
                "maxResults": _MAX_BATCH_SIZE,
                "returnData": True,
                "returnMetadata": True,
            }
            if next_token:
                params["nextToken"] = next_token
            resp = self._client.list_vectors(**params)
            for v in resp.get("vectors", []):
                key = v.get("key", "")
                emb = v.get("data", {}).get("float32", [])
                meta = dict(v.get("metadata", {}) or {})
                records.append(VectorRecord(key=key, embedding=emb, metadata=meta))
            next_token = resp.get("nextToken")
            if not next_token:
                break
        return records


# ---------------------------------------------------------------------------
# Internal math
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1]."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0))
