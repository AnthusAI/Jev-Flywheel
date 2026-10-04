"""
Provider-backed AI utilities for Biblicus.
"""

from __future__ import annotations

from .embeddings import generate_embeddings, generate_embeddings_batch
from .llm import generate_completion
from .models import AiProvider, EmbeddingsClientConfig, LlmClientConfig

__all__ = [
    "AiProvider",
    "EmbeddingsClientConfig",
    "LlmClientConfig",
    "generate_completion",
    "generate_embeddings",
    "generate_embeddings_batch",
]
