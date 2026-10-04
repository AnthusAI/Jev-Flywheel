"""
Public interface for the Biblicus Context Engine.
"""

from .assembler import ContextAssembler, ContextAssemblyResult
from .compaction import BaseCompactor, CompactionRequest, SummaryCompactor, TruncateCompactor
from .models import (
    AssistantMessageSpec,
    CompactorDeclaration,
    ContextBudgetSpec,
    ContextDeclaration,
    ContextExpansionSpec,
    ContextInsertSpec,
    ContextMessageSpec,
    ContextPackBudgetSpec,
    ContextPackSpec,
    ContextPolicySpec,
    ContextRetrieverRequest,
    ContextTemplateSpec,
    CorpusDeclaration,
    HistoryInsertSpec,
    RetrieverDeclaration,
    SystemMessageSpec,
    UserMessageSpec,
)
from .retrieval import retrieve_context_pack

__all__ = [
    "ContextAssembler",
    "ContextAssemblyResult",
    "BaseCompactor",
    "CompactionRequest",
    "SummaryCompactor",
    "TruncateCompactor",
    "ContextBudgetSpec",
    "ContextDeclaration",
    "ContextExpansionSpec",
    "ContextInsertSpec",
    "ContextMessageSpec",
    "ContextPackBudgetSpec",
    "ContextPackSpec",
    "ContextPolicySpec",
    "ContextRetrieverRequest",
    "ContextTemplateSpec",
    "CorpusDeclaration",
    "RetrieverDeclaration",
    "CompactorDeclaration",
    "HistoryInsertSpec",
    "SystemMessageSpec",
    "UserMessageSpec",
    "AssistantMessageSpec",
    "retrieve_context_pack",
]
