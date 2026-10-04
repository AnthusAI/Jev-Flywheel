"""
Biblicus public package interface.
"""

from .context_engine import (
    ContextAssembler,
    ContextBudgetSpec,
    ContextDeclaration,
    ContextExpansionSpec,
    ContextPackBudgetSpec,
    ContextPackSpec,
    ContextPolicySpec,
    ContextRetrieverRequest,
    retrieve_context_pack,
)
from .corpus import Corpus
from .knowledge_base import KnowledgeBase
from .models import (
    ConfigurationManifest,
    CorpusConfig,
    Evidence,
    IngestResult,
    QueryBudget,
    RetrievalResult,
    RetrievalSnapshot,
)
from .workflow import (
    Plan,
    Task,
    build_default_handler_registry,
    build_plan_for_extract,
    build_plan_for_index,
    build_plan_for_load,
    build_plan_for_query,
)

__all__ = [
    "__version__",
    "ContextAssembler",
    "ContextBudgetSpec",
    "ContextDeclaration",
    "ContextExpansionSpec",
    "ContextPackBudgetSpec",
    "ContextPackSpec",
    "ContextPolicySpec",
    "ContextRetrieverRequest",
    "retrieve_context_pack",
    "Corpus",
    "CorpusConfig",
    "Evidence",
    "IngestResult",
    "KnowledgeBase",
    "QueryBudget",
    "ConfigurationManifest",
    "RetrievalResult",
    "RetrievalSnapshot",
    "Plan",
    "Task",
    "build_default_handler_registry",
    "build_plan_for_extract",
    "build_plan_for_index",
    "build_plan_for_load",
    "build_plan_for_query",
]

__version__ = "1.11.3"
