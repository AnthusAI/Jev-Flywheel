"""
Reinforcement Memory — semantic topic memory for timestamped text streams.

Public API
----------
Core engine::

    from biblicus.analysis.reinforcement_memory import ReinforcementMemory

Data models::

    from biblicus.analysis.reinforcement_memory import (
        TimestampedText,
        AnalysisResult,
        TopicResult,
        ExemplarRecord,
        VectorRecord,
        QueryResult,
    )

Vector stores::

    from biblicus.analysis.reinforcement_memory import (
        VectorStore,
        LocalVectorStore,
        S3VectorStore,
    )

Embedding helpers::

    from biblicus.analysis.reinforcement_memory import (
        EmbedFn,
        hash_embedder,
        sentence_transformer_embedder,
        dspy_embedder,
        S3EmbeddingCache,
        LocalEmbeddingCache,
    )

LLM helpers::

    from biblicus.analysis.reinforcement_memory import (
        LabelFn,
        CausalFn,
        SynthesisFn,
        dspy_labeler,
        dspy_causal,
        dspy_synthesizer,
        bedrock_labeler,
        bedrock_causal,
        bedrock_synthesizer,
    )
"""

from ._embedding import (
    EmbedFn,
    LocalEmbeddingCache,
    S3EmbeddingCache,
    dspy_embedder,
    hash_embedder,
    sentence_transformer_embedder,
)
from ._engine import ReinforcementMemory
from ._llm import (
    CausalFn,
    LabelFn,
    SynthesisFn,
    bedrock_causal,
    bedrock_labeler,
    bedrock_synthesizer,
    dspy_causal,
    dspy_labeler,
    dspy_synthesizer,
)
from ._models import (
    AnalysisResult,
    ExemplarRecord,
    QueryResult,
    TimestampedText,
    TopicResult,
    VectorRecord,
)
from ._vector_store import LocalVectorStore, S3VectorStore, VectorStore

__all__ = [
    # Engine
    "ReinforcementMemory",
    # Models
    "TimestampedText",
    "AnalysisResult",
    "TopicResult",
    "ExemplarRecord",
    "VectorRecord",
    "QueryResult",
    # Vector stores
    "VectorStore",
    "LocalVectorStore",
    "S3VectorStore",
    # Embedding
    "EmbedFn",
    "hash_embedder",
    "sentence_transformer_embedder",
    "dspy_embedder",
    "S3EmbeddingCache",
    "LocalEmbeddingCache",
    # LLM
    "LabelFn",
    "CausalFn",
    "SynthesisFn",
    "dspy_labeler",
    "dspy_causal",
    "dspy_synthesizer",
    "bedrock_labeler",
    "bedrock_causal",
    "bedrock_synthesizer",
]
