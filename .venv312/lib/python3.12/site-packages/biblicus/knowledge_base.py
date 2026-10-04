"""
High-level knowledge base workflow for turnkey usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .context import (
    ContextPack,
    ContextPackPolicy,
    TokenBudget,
    build_context_pack,
    fit_context_pack_to_token_budget,
)
from .corpus import Corpus
from .models import QueryBudget, RetrievalResult, RetrievalSnapshot
from .retrievers import get_retriever


class KnowledgeBaseDefaults(BaseModel):
    """
    Default configuration for a knowledge base workflow.

    :ivar retriever_id: Retriever identifier to use for retrieval.
    :vartype retriever_id: str
    :ivar configuration_name: Human-readable retrieval configuration name.
    :vartype configuration_name: str
    :ivar query_budget: Default query budget to apply to retrieval.
    :vartype query_budget: QueryBudget
    :ivar tags: Tags to apply when importing the folder.
    :vartype tags: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    retriever_id: str = Field(default="scan", min_length=1)
    configuration_name: str = Field(default="Knowledge base", min_length=1)
    query_budget: QueryBudget = Field(
        default_factory=lambda: QueryBudget(
            max_total_items=5,
            maximum_total_characters=2000,
            max_items_per_source=None,
        )
    )
    tags: List[str] = Field(default_factory=list)


@dataclass
class KnowledgeBase:
    """
    High-level knowledge base wrapper for turnkey workflows.

    :ivar corpus: Corpus instance that stores the ingested items.
    :vartype corpus: Corpus
    :ivar retriever_id: Retriever identifier used for retrieval.
    :vartype retriever_id: str
    :ivar snapshot: Retrieval snapshot manifest associated with the knowledge base.
    :vartype snapshot: RetrievalSnapshot
    :ivar defaults: Default configuration used for this knowledge base.
    :vartype defaults: KnowledgeBaseDefaults
    """

    corpus: Corpus
    retriever_id: str
    snapshot: RetrievalSnapshot
    defaults: KnowledgeBaseDefaults
    _temp_dir: Optional[TemporaryDirectory]

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        retriever_id: Optional[str] = None,
        configuration_name: Optional[str] = None,
        query_budget: Optional[QueryBudget] = None,
        tags: Optional[Sequence[str]] = None,
        corpus_root: Optional[str | Path] = None,
    ) -> "KnowledgeBase":
        """
        Build a knowledge base from a folder of files.

        :param folder: Folder containing source files.
        :type folder: str or Path
        :param retriever_id: Optional retriever identifier override.
        :type retriever_id: str or None
        :param configuration_name: Optional configuration name override.
        :type configuration_name: str or None
        :param query_budget: Optional query budget override.
        :type query_budget: QueryBudget or None
        :param tags: Optional tags to apply during import.
        :type tags: Sequence[str] or None
        :param corpus_root: Optional corpus root override. Must contain the source folder.
        :type corpus_root: str or Path or None
        :return: Knowledge base instance.
        :rtype: KnowledgeBase
        :raises FileNotFoundError: If the folder does not exist.
        :raises NotADirectoryError: If the folder is not a directory.
        """
        source_root = Path(folder).resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Knowledge base folder does not exist: {source_root}")
        if not source_root.is_dir():
            raise NotADirectoryError(f"Knowledge base folder is not a directory: {source_root}")

        defaults = KnowledgeBaseDefaults()
        resolved_retriever_id = retriever_id or defaults.retriever_id
        resolved_configuration_name = configuration_name or defaults.configuration_name
        resolved_query_budget = query_budget or defaults.query_budget
        resolved_tags = list(tags) if tags is not None else defaults.tags

        temp_dir: Optional[TemporaryDirectory] = None
        if corpus_root is None:
            corpus_root_path = source_root
        else:
            corpus_root_path = Path(corpus_root).resolve()
            if not source_root.is_relative_to(corpus_root_path):
                raise ValueError(
                    "Knowledge base source folder must live inside the corpus root."
                )

        if (corpus_root_path / "metadata" / "config.json").is_file():
            corpus = Corpus.open(corpus_root_path)
        else:
            corpus = Corpus.init(corpus_root_path)
        corpus.import_tree(source_root, tags=resolved_tags)

        retriever = get_retriever(resolved_retriever_id)
        snapshot = retriever.build_snapshot(
            corpus, configuration_name=resolved_configuration_name, configuration={}
        )

        return cls(
            corpus=corpus,
            retriever_id=resolved_retriever_id,
            snapshot=snapshot,
            defaults=KnowledgeBaseDefaults(
                retriever_id=resolved_retriever_id,
                configuration_name=resolved_configuration_name,
                query_budget=resolved_query_budget,
                tags=resolved_tags,
            ),
            _temp_dir=temp_dir,
        )

    def query(self, query_text: str, *, budget: Optional[QueryBudget] = None) -> RetrievalResult:
        """
        Query the knowledge base for evidence.

        :param query_text: Query text to execute.
        :type query_text: str
        :param budget: Optional budget override.
        :type budget: QueryBudget or None
        :return: Retrieval result containing evidence.
        :rtype: RetrievalResult
        """
        retriever = get_retriever(self.retriever_id)
        resolved_budget = budget or self.defaults.query_budget
        return retriever.query(
            self.corpus,
            snapshot=self.snapshot,
            query_text=query_text,
            budget=resolved_budget,
        )

    def context_pack(
        self,
        result: RetrievalResult,
        *,
        join_with: str = "\n\n",
        max_tokens: Optional[int] = None,
    ) -> ContextPack:
        """
        Build a context pack from a retrieval result.

        :param result: Retrieval result to convert into context.
        :type result: RetrievalResult
        :param join_with: Join string for evidence blocks.
        :type join_with: str
        :param max_tokens: Optional token budget for the context pack.
        :type max_tokens: int or None
        :return: Context pack text and metadata.
        :rtype: ContextPack
        """
        policy = ContextPackPolicy(join_with=join_with)
        context_pack = build_context_pack(result, policy=policy)
        if max_tokens is None:
            return context_pack
        return fit_context_pack_to_token_budget(
            context_pack,
            policy=policy,
            token_budget=TokenBudget(max_tokens=max_tokens),
        )
