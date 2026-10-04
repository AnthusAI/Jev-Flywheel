"""
Context retrieval helpers for the Biblicus Context Engine.
"""

from __future__ import annotations

from typing import Any, Optional

from biblicus.context import (
    ContextPack,
    ContextPackPolicy,
    TokenBudget,
    build_context_pack,
    fit_context_pack_to_token_budget,
)
from biblicus.corpus import Corpus
from biblicus.models import QueryBudget, RetrievalSnapshot
from biblicus.retrievers import get_retriever

from .models import ContextRetrieverRequest


def _resolve_snapshot(
    corpus: Corpus,
    *,
    retriever_id: str,
    snapshot_id: Optional[str],
    configuration_name: Optional[str],
    configuration: Optional[dict[str, Any]],
) -> RetrievalSnapshot:
    if snapshot_id:
        return corpus.load_snapshot(snapshot_id)

    latest_snapshot_id = corpus.latest_snapshot_id
    if latest_snapshot_id:
        candidate = corpus.load_snapshot(latest_snapshot_id)
        if candidate.configuration.retriever_id == retriever_id:
            return candidate

    if configuration is None:
        raise ValueError(
            "No retrieval snapshot available for the requested retriever. "
            "Provide snapshot_id or configuration to build one."
        )

    retriever = get_retriever(retriever_id)
    resolved_name = configuration_name or f"Context pack ({retriever_id})"
    return retriever.build_snapshot(
        corpus,
        configuration_name=resolved_name,
        configuration=configuration,
    )


def retrieve_context_pack(
    *,
    request: ContextRetrieverRequest,
    corpus: Corpus,
    retriever_id: str,
    snapshot_id: Optional[str] = None,
    configuration_name: Optional[str] = None,
    configuration: Optional[dict[str, Any]] = None,
    join_with: str = "\n\n",
    max_items_per_source: Optional[int] = None,
    include_metadata: bool = False,
    metadata_fields: Optional[list[str]] = None,
) -> ContextPack:
    """
    Retrieve a context pack using a Biblicus retriever.

    :param request: Context retrieval request.
    :type request: biblicus.context_engine.ContextRetrieverRequest
    :param corpus: Corpus instance to query.
    :type corpus: biblicus.corpus.Corpus
    :param retriever_id: Retrieval retriever identifier.
    :type retriever_id: str
    :param snapshot_id: Optional retrieval snapshot identifier.
    :type snapshot_id: str or None
    :param configuration_name: Optional configuration name for snapshot builds.
    :type configuration_name: str or None
    :param configuration: Optional retriever configuration.
    :type configuration: dict[str, Any] or None
    :param join_with: Separator between context pack blocks.
    :type join_with: str
    :param max_items_per_source: Optional cap per source.
    :type max_items_per_source: int or None
    :param include_metadata: Whether to include metadata in context blocks.
    :type include_metadata: bool
    :param metadata_fields: Optional metadata fields to include in context blocks.
    :type metadata_fields: list[str] or None
    :return: Context pack derived from retrieval results.
    :rtype: biblicus.context.ContextPack
    :raises ValueError: If no compatible retrieval snapshot is available.
    """
    snapshot = _resolve_snapshot(
        corpus,
        retriever_id=retriever_id,
        snapshot_id=snapshot_id,
        configuration_name=configuration_name,
        configuration=configuration,
    )

    maximum_total_characters = request.maximum_total_characters
    if maximum_total_characters is None and request.max_tokens is not None:
        maximum_total_characters = int(request.max_tokens * 4)

    budget = QueryBudget(
        max_total_items=request.limit,
        offset=request.offset,
        maximum_total_characters=maximum_total_characters,
        max_items_per_source=max_items_per_source,
    )
    retriever = get_retriever(retriever_id)
    result = retriever.query(
        corpus,
        snapshot=snapshot,
        query_text=request.query,
        budget=budget,
    )
    policy = ContextPackPolicy(
        join_with=join_with,
        include_metadata=include_metadata,
        metadata_fields=metadata_fields,
    )
    context_pack = build_context_pack(result, policy=policy)
    if request.max_tokens is None:
        return context_pack

    return fit_context_pack_to_token_budget(
        context_pack,
        policy=policy,
        token_budget=TokenBudget(max_tokens=int(request.max_tokens)),
    )
