"""
Context pack building for Biblicus.

A context pack is the text that your application sends to a large language model.
Biblicus produces a context pack from structured retrieval results so that evidence remains a
stable contract while context formatting remains an explicit policy surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import Evidence, RetrievalResult


class ContextPackPolicy(BaseModel):
    """
    Policy that controls how evidence becomes context pack text.

    :ivar join_with: Separator inserted between evidence text blocks.
    :vartype join_with: str
    :ivar ordering: Evidence ordering policy (rank, score, or source).
    :vartype ordering: str
    :ivar include_metadata: Whether to include evidence metadata lines in each block.
    :vartype include_metadata: bool
    :ivar metadata_fields: Optional evidence metadata fields to include.
    :vartype metadata_fields: list[str] or None
    """

    model_config = ConfigDict(extra="forbid")

    join_with: str = Field(default="\n\n")
    ordering: str = Field(default="rank", min_length=1)
    include_metadata: bool = Field(default=False)
    metadata_fields: Optional[List[str]] = None


class ContextPack(BaseModel):
    """
    Context pack derived from retrieval evidence.

    :ivar text: Context pack text suitable for inclusion in a model call.
    :vartype text: str
    :ivar evidence_count: Number of evidence blocks included in the context pack.
    :vartype evidence_count: int
    :ivar blocks: Structured blocks that produced the context pack.
    :vartype blocks: list[ContextPackBlock]
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_count: int = Field(ge=0)
    blocks: List["ContextPackBlock"] = Field(default_factory=list)


class ContextPackBlock(BaseModel):
    """
    A single context pack block derived from one evidence item.

    :ivar evidence_item_id: Item identifier that produced this block.
    :vartype evidence_item_id: str
    :ivar text: Text included in this block.
    :vartype text: str
    :ivar metadata: Optional metadata included with the block.
    :vartype metadata: dict[str, object] or None
    """

    model_config = ConfigDict(extra="forbid")

    evidence_item_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: Optional[Dict[str, object]] = None


class TokenCounter(BaseModel):
    """
    Token counter configuration for token budget fitting.

    This is a lightweight model wrapper so token fitting remains explicit and testable even when
    the underlying tokenizer is provided by an optional dependency.

    :ivar tokenizer_id: Tokenizer identifier (for example, naive-whitespace).
    :vartype tokenizer_id: str
    """

    model_config = ConfigDict(extra="forbid")

    tokenizer_id: str = Field(default="naive-whitespace", min_length=1)


class TokenBudget(BaseModel):
    """
    Token budget for a context pack.

    :ivar max_tokens: Maximum tokens permitted for the final context pack text.
    :vartype max_tokens: int
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(ge=1)


class CharacterBudget(BaseModel):
    """
    Character budget for a context pack.

    :ivar max_characters: Maximum characters permitted for the final context pack text.
    :vartype max_characters: int
    """

    model_config = ConfigDict(extra="forbid")

    max_characters: int = Field(ge=1)


def build_context_pack(result: RetrievalResult, *, policy: ContextPackPolicy) -> ContextPack:
    """
    Build a context pack from a retrieval result using an explicit policy.

    :param result: Retrieval result containing ranked evidence.
    :type result: RetrievalResult
    :param policy: Policy controlling how evidence text is joined.
    :type policy: ContextPackPolicy
    :return: Context pack containing concatenated evidence text.
    :rtype: ContextPack
    """
    selected_blocks: List[ContextPackBlock] = []
    for evidence in _order_evidence(result.evidence, policy=policy):
        if not isinstance(evidence.text, str):
            continue
        trimmed_text = evidence.text.strip()
        if not trimmed_text:
            continue
        metadata = (
            _metadata_for_evidence(evidence, policy=policy) if policy.include_metadata else None
        )
        block_text = _format_block_text(trimmed_text, metadata=metadata)
        selected_blocks.append(
            ContextPackBlock(
                evidence_item_id=evidence.item_id,
                text=block_text,
                metadata=metadata,
            )
        )

    return ContextPack(
        text=policy.join_with.join([block.text for block in selected_blocks]),
        evidence_count=len(selected_blocks),
        blocks=selected_blocks,
    )


def count_tokens(text: str, *, tokenizer_id: str) -> int:
    """
    Count tokens in a text using a tokenizer identifier.

    The default tokenizer is naive-whitespace, which counts whitespace-separated tokens.

    :param text: Text payload to count.
    :type text: str
    :param tokenizer_id: Tokenizer identifier.
    :type tokenizer_id: str
    :return: Token count.
    :rtype: int
    :raises KeyError: If the tokenizer identifier is unknown.
    """
    tokenizers = {
        "naive-whitespace": lambda value: len([token for token in value.split() if token]),
    }
    tokenizer = tokenizers[tokenizer_id]
    return int(tokenizer(text))


def fit_context_pack_to_token_budget(
    context_pack: ContextPack,
    *,
    policy: ContextPackPolicy,
    token_budget: TokenBudget,
    token_counter: Optional[TokenCounter] = None,
) -> ContextPack:
    """
    Fit a context pack to a token budget by dropping trailing blocks.

    This function is deterministic. It never rewrites block text. It only removes blocks from the
    end of the block list until the token budget is met.

    :param context_pack: Context pack to fit.
    :type context_pack: ContextPack
    :param policy: Policy controlling how blocks are joined into text.
    :type policy: ContextPackPolicy
    :param token_budget: Token budget to enforce.
    :type token_budget: TokenBudget
    :param token_counter: Optional token counter configuration.
    :type token_counter: TokenCounter or None
    :return: Fitted context pack.
    :rtype: ContextPack
    """
    token_counter = token_counter or TokenCounter()
    remaining_blocks: List[ContextPackBlock] = list(context_pack.blocks)

    while remaining_blocks:
        candidate_text = policy.join_with.join([block.text for block in remaining_blocks])
        candidate_tokens = count_tokens(candidate_text, tokenizer_id=token_counter.tokenizer_id)
        if candidate_tokens <= token_budget.max_tokens:
            return ContextPack(
                text=candidate_text,
                evidence_count=len(remaining_blocks),
                blocks=remaining_blocks,
            )
        remaining_blocks = remaining_blocks[:-1]

    return ContextPack(text="", evidence_count=0, blocks=[])


def fit_context_pack_to_character_budget(
    context_pack: ContextPack,
    *,
    policy: ContextPackPolicy,
    character_budget: CharacterBudget,
) -> ContextPack:
    """
    Fit a context pack to a character budget by dropping trailing blocks.

    :param context_pack: Context pack to fit.
    :type context_pack: ContextPack
    :param policy: Policy controlling how blocks are joined into text.
    :type policy: ContextPackPolicy
    :param character_budget: Character budget to enforce.
    :type character_budget: CharacterBudget
    :return: Fitted context pack.
    :rtype: ContextPack
    """
    remaining_blocks: List[ContextPackBlock] = list(context_pack.blocks)
    max_characters = character_budget.max_characters

    while remaining_blocks:
        candidate_text = policy.join_with.join([block.text for block in remaining_blocks])
        if len(candidate_text) <= max_characters:
            return ContextPack(
                text=candidate_text,
                evidence_count=len(remaining_blocks),
                blocks=remaining_blocks,
            )
        remaining_blocks = remaining_blocks[:-1]

    return ContextPack(text="", evidence_count=0, blocks=[])


def _order_evidence(
    evidence: List[Evidence],
    *,
    policy: ContextPackPolicy,
) -> List[Evidence]:
    """
    Order evidence items according to the context pack policy.

    :param evidence: Evidence list to order.
    :type evidence: list[Evidence]
    :param policy: Context pack policy.
    :type policy: ContextPackPolicy
    :return: Ordered evidence list.
    :rtype: list[Evidence]
    """
    if policy.ordering == "rank":
        return sorted(evidence, key=lambda item: (item.rank, item.item_id))
    if policy.ordering == "score":
        return sorted(evidence, key=lambda item: (-item.score, item.item_id))
    if policy.ordering == "source":
        return sorted(
            evidence,
            key=lambda item: (
                item.source_uri or item.item_id,
                -item.score,
                item.item_id,
            ),
        )
    raise ValueError(f"Unknown context pack ordering: {policy.ordering}")


def _metadata_for_evidence(
    evidence: Evidence,
    *,
    policy: ContextPackPolicy,
) -> Dict[str, object]:
    """
    Build metadata for a context pack block.

    :param evidence: Evidence item to describe.
    :type evidence: Evidence
    :return: Metadata mapping.
    :rtype: dict[str, object]
    """
    metadata = {
        "item_id": evidence.item_id,
        "source_uri": evidence.source_uri or "none",
        "score": evidence.score,
        "stage": evidence.stage,
    }
    extra = evidence.metadata or {}
    if policy.metadata_fields is not None:
        extra = {key: extra.get(key) for key in policy.metadata_fields if key in extra}
    for key, value in extra.items():
        if key not in metadata:
            metadata[key] = value
    return metadata


def _format_block_text(text: str, *, metadata: Optional[Dict[str, object]]) -> str:
    """
    Format a context pack block text with optional metadata.

    :param text: Evidence text.
    :type text: str
    :param metadata: Optional metadata mapping.
    :type metadata: dict[str, object] or None
    :return: Formatted block text.
    :rtype: str
    """
    if not metadata:
        return text
    ordered_keys = ["item_id", "source_uri", "score", "stage"]
    metadata_lines = [f"{key}: {metadata[key]}" for key in ordered_keys if key in metadata]
    for key in sorted(metadata.keys()):
        if key in ordered_keys:
            continue
        metadata_lines.append(f"{key}: {metadata[key]}")
    metadata_text = "\n".join(metadata_lines)
    return f"{metadata_text}\n{text}"
