"""
Provider-backed text embeddings.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Sequence

from .models import EmbeddingsClientConfig


def _require_dspy_embedder():
    try:
        import dspy
    except ImportError as import_error:
        raise ValueError(
            "DSPy backend requires an optional dependency. "
            'Install it with pip install "biblicus[dspy]".'
        ) from import_error
    if not hasattr(dspy, "Embedder"):
        raise ValueError(
            "DSPy backend requires an optional dependency with Embedder support. "
            'Install it with pip install "biblicus[dspy]".'
        )
    return dspy


def generate_embeddings(*, client: EmbeddingsClientConfig, text: str) -> List[float]:
    """
    Generate a single embedding vector.

    :param client: Embeddings client configuration.
    :type client: biblicus.ai.models.EmbeddingsClientConfig
    :param text: Input text to embed.
    :type text: str
    :return: Embedding vector.
    :rtype: list[float]
    """
    vectors = generate_embeddings_batch(client=client, texts=[text])
    return vectors[0]


def _chunks(texts: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(texts[idx : idx + batch_size]) for idx in range(0, len(texts), batch_size)]


def _normalize_embeddings(embeddings: Any) -> List[List[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if isinstance(embeddings, list) and embeddings and not isinstance(embeddings[0], list):
        return [[float(value) for value in embeddings]]
    return [[float(value) for value in row] for row in embeddings]


def generate_embeddings_batch(
    *, client: EmbeddingsClientConfig, texts: Sequence[str]
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.

    The implementation performs batched requests and can run requests concurrently.

    :param client: Embeddings client configuration.
    :type client: biblicus.ai.models.EmbeddingsClientConfig
    :param texts: Text inputs to embed.
    :type texts: Sequence[str]
    :return: Embedding vectors in input order.
    :rtype: list[list[float]]
    :raises ValueError: If required dependencies or credentials are missing.
    """
    if not texts:
        return []

    dspy = _require_dspy_embedder()

    model = client.litellm_model()
    request_kwargs = client.build_litellm_kwargs()

    items = list(texts)
    if len(items) == 1:
        embedder = dspy.Embedder(
            model,
            batch_size=1,
            caching=False,
            **request_kwargs,
        )
        embeddings = embedder(items[0])
        return _normalize_embeddings(embeddings)

    batches = _chunks(items, client.batch_size)
    results: List[List[List[float]]] = [None for _ in range(len(batches))]  # type: ignore[list-item]

    def _embed_batch(batch_texts: List[str]) -> List[List[float]]:
        embedder = dspy.Embedder(
            model,
            batch_size=len(batch_texts),
            caching=False,
            **request_kwargs,
        )
        embeddings = embedder(batch_texts)
        return _normalize_embeddings(embeddings)

    with ThreadPoolExecutor(max_workers=client.parallelism) as executor:
        futures = {executor.submit(_embed_batch, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    flattened: List[List[float]] = []
    for batch_vectors in results:
        for vector in batch_vectors:
            flattened.append(vector)
    return flattened
