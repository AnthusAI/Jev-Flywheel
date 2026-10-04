"""
Compaction utilities for Context Engine assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompactionRequest:
    """
    Request payload for compaction.

    :ivar text: Text to compact.
    :vartype text: str
    :ivar max_tokens: Maximum token budget.
    :vartype max_tokens: int
    """

    text: str
    max_tokens: int


class BaseCompactor:
    """
    Base class for compaction strategies.

    Subclasses implement ``compact`` to return a shorter string that fits the
    requested budget.
    """

    def compact(self, request: CompactionRequest) -> str:
        """
        Compact text to fit within the requested token budget.

        :param request: Compaction request with text and budget.
        :type request: CompactionRequest
        :return: Compacted text.
        :rtype: str
        """
        raise NotImplementedError


class TruncateCompactor(BaseCompactor):
    """
    Simple truncation compactor (token-based).
    """

    def compact(self, request: CompactionRequest) -> str:
        """
        Compact by truncating to the maximum token count.

        :param request: Compaction request with text and budget.
        :type request: CompactionRequest
        :return: Truncated text.
        :rtype: str
        """
        tokens = request.text.split()
        if len(tokens) <= request.max_tokens:
            return request.text
        return " ".join(tokens[: request.max_tokens])


class SummaryCompactor(BaseCompactor):
    """
    Simple sentence-first compactor (deterministic).
    """

    def compact(self, request: CompactionRequest) -> str:
        """
        Compact by selecting the first sentence within the budget.

        :param request: Compaction request with text and budget.
        :type request: CompactionRequest
        :return: Compacted text.
        :rtype: str
        """
        sentences = _split_sentences(request.text)
        if not sentences:
            return request.text

        compacted = sentences[0].strip()
        tokens = compacted.split()
        if len(tokens) > request.max_tokens:
            return " ".join(tokens[: request.max_tokens])
        return compacted


def build_compactor(config: dict[str, Any]) -> BaseCompactor:
    """
    Build a compactor instance from configuration.

    :param config: Compactor configuration payload.
    :type config: dict[str, Any]
    :return: Compactor instance.
    :rtype: BaseCompactor
    :raises ValueError: If the compactor type is unknown.
    """
    strategy = config.get("type", "truncate")
    if strategy == "truncate":
        return TruncateCompactor()
    if strategy == "summary":
        return SummaryCompactor()
    raise ValueError(f"Unknown compactor type: {strategy}")


def _split_sentences(text: str) -> list[str]:
    return [segment for segment in text.split(". ") if segment]
