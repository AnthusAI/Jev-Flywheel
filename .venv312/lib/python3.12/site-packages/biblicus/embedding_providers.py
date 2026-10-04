"""
Embedding provider interfaces for retrieval backends.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingProvider(ABC):
    """
    Interface for producing dense embedding vectors from text.

    :ivar provider_id: Provider identifier.
    :vartype provider_id: str
    """

    provider_id: str

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        :param texts: Text inputs.
        :type texts: Sequence[str]
        :return: 2D float array with shape (len(texts), dimensions).
        :rtype: numpy.ndarray
        """
        raise NotImplementedError


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic embedding provider for tests and demos.

    The output vectors are stable across runs and require no external services.
    """

    provider_id = "hash-embedding"

    def __init__(self, *, dimensions: int, seed: str = "biblicus") -> None:
        self._dimensions = int(dimensions)
        self._seed = str(seed)
        if self._dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a batch of texts deterministically.

        :param texts: Text inputs.
        :type texts: Sequence[str]
        :return: Normalized embedding matrix.
        :rtype: numpy.ndarray
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self._dimensions), dtype=np.float32)

        vectors = np.zeros((len(items), self._dimensions), dtype=np.float32)
        for row_index, text in enumerate(items):
            vectors[row_index] = self._hash_to_vector(text)
        return _l2_normalize_rows(vectors)

    def _hash_to_vector(self, text: str) -> np.ndarray:
        output = np.empty((self._dimensions,), dtype=np.float32)
        remaining = self._dimensions
        offset = 0
        counter = 0
        while remaining > 0:
            digest = hashlib.sha256(f"{self._seed}:{counter}:{text}".encode("utf-8")).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            raw = (raw / 255.0) * 2.0 - 1.0
            take = min(remaining, raw.shape[0])
            output[offset : offset + take] = raw[:take]
            remaining -= take
            offset += take
            counter += 1
        return output


class EmbeddingProviderConfig(BaseModel):
    """
    Configuration for embedding provider selection.

    :ivar provider_id: Provider identifier.
    :vartype provider_id: str
    :ivar dimensions: Dimensionality of produced vectors.
    :vartype dimensions: int
    :ivar seed: Optional deterministic seed for test providers.
    :vartype seed: str or None
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    seed: Optional[str] = None

    def build_provider(self) -> EmbeddingProvider:
        """
        Build an embedding provider instance from this configuration.

        :return: Embedding provider instance.
        :rtype: EmbeddingProvider
        :raises ValueError: If the provider identifier is unknown.
        """
        if self.provider_id == HashEmbeddingProvider.provider_id:
            return HashEmbeddingProvider(dimensions=self.dimensions, seed=self.seed or "biblicus")
        raise ValueError(f"Unknown embedding provider_id: {self.provider_id!r}")
