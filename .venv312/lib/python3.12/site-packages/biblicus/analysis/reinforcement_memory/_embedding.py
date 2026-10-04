"""
Embedding helpers for Reinforcement Memory.

Defines the :data:`EmbedFn` callable type and provides factory functions
that return it:

- :func:`hash_embedder` — deterministic hash-based embeddings, no external
  services.  Useful for tests and demos.
- :func:`sentence_transformer_embedder` — sentence-transformers with optional
  S3 or local caching.  Requires ``sentence-transformers``.
- :func:`dspy_embedder` — Biblicus's DSPy-based embedding backend.  Requires
  the ``dspy`` optional dependency group.

Also provides :class:`EmbeddingCacheProtocol`, :class:`S3EmbeddingCache`, and
:class:`LocalEmbeddingCache` for the caching layer used by
:func:`sentence_transformer_embedder`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Callable, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for an embedding function accepted by :class:`ReinforcementMemory`.
EmbedFn = Callable[[List[str]], np.ndarray]

_DEFAULT_MODEL_ID = "all-MiniLM-L6-v2"
_DEFAULT_PREPROCESSING_VERSION = "1"


# ---------------------------------------------------------------------------
# Embedding cache protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingCacheProtocol(Protocol):
    """Protocol for an embedding cache backend."""

    def get(self, model_id: str, key: str) -> Optional[np.ndarray]:
        """Return cached embedding or None on miss."""
        pass

    def put(self, model_id: str, key: str, embedding: np.ndarray) -> None:
        """Store an embedding (non-fatal on failure)."""
        pass


def _normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase for deterministic cache keys."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _cache_key(text: str, model_id: str, version: str) -> str:
    """Return SHA-256 cache key for a text + model + version triple."""
    payload = f"{_normalize_text(text)}|{model_id}|{version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _s3_path(model_id: str, key: str) -> str:
    prefix = key[:2] if len(key) >= 2 else "00"
    return f"embeddings/{model_id}/{prefix}/{key}.json"


class S3EmbeddingCache:
    """
    S3-backed embedding cache.

    Cache key: ``SHA-256(normalize(text)|model_id|version)``
    S3 path: ``embeddings/{model_id}/{key_prefix}/{key}.json``

    :param bucket_name: S3 bucket for cached embeddings.
    :param s3_client: Optional pre-built boto3 S3 client.
    """

    def __init__(
        self,
        bucket_name: str,
        s3_client=None,
    ) -> None:
        """Initialise the S3 cache."""
        self.bucket_name = bucket_name
        if s3_client is not None:
            self._s3 = s3_client
        else:
            import boto3

            self._s3 = boto3.client("s3")

    def get(self, model_id: str, key: str) -> Optional[np.ndarray]:
        """Return cached embedding or None on miss."""
        try:
            path = _s3_path(model_id, key)
            resp = self._s3.get_object(Bucket=self.bucket_name, Key=path)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            vec = data.get("embedding")
            return np.array(vec, dtype=np.float32) if vec is not None else None
        except Exception:
            return None

    def put(self, model_id: str, key: str, embedding: np.ndarray) -> None:
        """Write embedding to S3 (non-fatal on failure)."""
        try:
            path = _s3_path(model_id, key)
            body = json.dumps({"embedding": embedding.tolist()}).encode("utf-8")
            self._s3.put_object(
                Bucket=self.bucket_name,
                Key=path,
                Body=body,
                ContentType="application/json",
            )
        except Exception as exc:
            logger.warning("S3EmbeddingCache put failed (non-fatal): %s", exc)


class LocalEmbeddingCache:
    """
    Filesystem-backed embedding cache.

    Cache files are stored as JSON under ``cache_dir/{model_id}/{prefix}/{key}.json``.

    :param cache_dir: Root directory for cached embeddings.
    """

    def __init__(self, cache_dir: str) -> None:
        """Initialise the local cache."""
        self._root = cache_dir

    def get(self, model_id: str, key: str) -> Optional[np.ndarray]:
        """Return cached embedding or None on miss."""
        path = self._path(model_id, key)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            vec = data.get("embedding")
            return np.array(vec, dtype=np.float32) if vec is not None else None
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            return None

    def put(self, model_id: str, key: str, embedding: np.ndarray) -> None:
        """Write embedding to disk (non-fatal on failure)."""
        path = self._path(model_id, key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"embedding": embedding.tolist()}, fh)
        except Exception as exc:
            logger.warning("LocalEmbeddingCache put failed (non-fatal): %s", exc)

    def _path(self, model_id: str, key: str) -> str:
        prefix = key[:2] if len(key) >= 2 else "00"
        safe_model = model_id.replace("/", "_")
        return os.path.join(self._root, safe_model, prefix, f"{key}.json")


# ---------------------------------------------------------------------------
# Embedding factory helpers
# ---------------------------------------------------------------------------


def hash_embedder(dimensions: int = 384) -> EmbedFn:
    """
    Return an :data:`EmbedFn` that produces deterministic hash-based embeddings.

    No external services or models required.  Suitable for tests and demos.

    :param dimensions: Output vector dimension.
    :return: Callable that maps a list of strings to an ndarray of shape
        ``(n, dimensions)``.
    """

    def _embed(texts: List[str]) -> np.ndarray:
        result = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dimensions).astype(np.float32)
            norm = np.linalg.norm(vec)
            result.append(vec / norm if norm > 0 else vec)
        return np.array(result, dtype=np.float32)

    return _embed


def sentence_transformer_embedder(
    model_id: str = _DEFAULT_MODEL_ID,
    cache: Optional[EmbeddingCacheProtocol] = None,
    preprocessing_version: str = _DEFAULT_PREPROCESSING_VERSION,
) -> EmbedFn:
    """
    Return an :data:`EmbedFn` backed by sentence-transformers.

    Results are cached via ``cache`` when provided, avoiding re-embedding
    identical texts across runs.

    Requires ``sentence-transformers`` (``pip install biblicus[sentence-transformers]``).

    :param model_id: Sentence-transformers model name (default
        ``"all-MiniLM-L6-v2"``).
    :param cache: Optional :class:`EmbeddingCacheProtocol` implementation.
        Use :class:`S3EmbeddingCache` for production or
        :class:`LocalEmbeddingCache` for local development.
    :param preprocessing_version: Version string included in the cache key.
        Increment this to invalidate the cache after preprocessing changes.
    :return: Callable that maps a list of strings to an ndarray of shape
        ``(n, dim)``.
    """
    _model_holder: list = []  # mutable container for lazy singleton

    def _load_model():
        if not _model_holder:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence_transformer_embedder requires sentence-transformers. "
                    'Install it with: pip install "biblicus[sentence-transformers]"'
                ) from exc
            _model_holder.append(SentenceTransformer(model_id))
        return _model_holder[0]

    def _embed(texts: List[str]) -> np.ndarray:
        keys = [_cache_key(t, model_id, preprocessing_version) for t in texts]
        result: list = [None] * len(texts)
        miss_indices: list = []
        miss_texts: list = []

        if cache is not None:
            for i, (text, key) in enumerate(zip(texts, keys)):
                cached = cache.get(model_id, key)
                if cached is not None:
                    result[i] = cached
                else:
                    miss_indices.append(i)
                    miss_texts.append(text)
        else:
            miss_indices = list(range(len(texts)))
            miss_texts = list(texts)

        if miss_texts:
            model = _load_model()
            embeddings = model.encode(miss_texts, convert_to_numpy=True)
            for idx, emb in zip(miss_indices, embeddings):
                arr = np.asarray(emb, dtype=np.float32)
                result[idx] = arr
                if cache is not None:
                    cache.put(model_id, keys[idx], arr)

        return np.array(result, dtype=np.float32)

    return _embed


def dspy_embedder(client) -> EmbedFn:
    """
    Return an :data:`EmbedFn` backed by Biblicus's DSPy embedding backend.

    :param client: :class:`~biblicus.ai.models.EmbeddingsClientConfig` instance.
    :return: Callable that maps a list of strings to an ndarray.
    """
    from biblicus.ai.embeddings import generate_embeddings_batch

    def _embed(texts: List[str]) -> np.ndarray:
        vectors = generate_embeddings_batch(client=client, texts=texts)
        return np.array(vectors, dtype=np.float32)

    return _embed
