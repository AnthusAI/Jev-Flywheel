"""
Chunking primitives for text retrieval backends.

Chunking converts a document-sized text string into smaller spans that can be embedded or indexed.
The chunk span offsets are expressed as character positions into the original text string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenSpan(BaseModel):
    """
    A token with its character span in a source string.

    :ivar token: Token text.
    :vartype token: str
    :ivar span_start: Inclusive start character offset.
    :vartype span_start: int
    :ivar span_end: Exclusive end character offset.
    :vartype span_end: int
    """

    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "TokenSpan":
        if self.span_end <= self.span_start:
            raise ValueError("token span_end must be greater than span_start")
        return self


class Tokenizer(ABC):
    """
    Interface for producing token spans from text.

    Tokenizers are used by token-window chunking strategies to convert token indices into
    stable character spans.

    :ivar tokenizer_id: Tokenizer identifier.
    :vartype tokenizer_id: str
    """

    tokenizer_id: str

    @abstractmethod
    def tokenize(self, text: str) -> List[TokenSpan]:
        """
        Tokenize a string and return spans for each token.

        :param text: Input text.
        :type text: str
        :return: Token spans.
        :rtype: list[TokenSpan]
        """
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizer that treats runs of non-whitespace characters as tokens.
    """

    tokenizer_id = "whitespace"

    def tokenize(self, text: str) -> List[TokenSpan]:
        """
        Tokenize a string by whitespace boundaries.

        :param text: Input text.
        :type text: str
        :return: Token spans for each non-whitespace token.
        :rtype: list[TokenSpan]
        """
        import re

        spans: List[TokenSpan] = []
        for match in re.finditer(r"\S+", text):
            spans.append(
                TokenSpan(
                    token=match.group(0),
                    span_start=int(match.start()),
                    span_end=int(match.end()),
                )
            )
        return spans


class TextChunk(BaseModel):
    """
    A chunk extracted from a larger text string.

    :ivar chunk_id: Stable chunk identifier within a build.
    :vartype chunk_id: int
    :ivar item_id: Source item identifier.
    :vartype item_id: str
    :ivar span_start: Inclusive start character offset.
    :vartype span_start: int
    :ivar span_end: Exclusive end character offset.
    :vartype span_end: int
    :ivar text: Chunk text.
    :vartype text: str
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(ge=0)
    item_id: str = Field(min_length=1)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _validate_span(self) -> "TextChunk":
        if self.span_end <= self.span_start:
            raise ValueError("chunk span_end must be greater than span_start")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("chunk text must be non-empty")
        return self


class Chunker(ABC):
    """
    Interface for converting text into chunks.

    :ivar chunker_id: Chunker identifier.
    :vartype chunker_id: str
    """

    chunker_id: str

    @abstractmethod
    def chunk_text(self, *, item_id: str, text: str, starting_chunk_id: int) -> List[TextChunk]:
        """
        Split a text string into chunks.

        :param item_id: Item identifier that produced the text.
        :type item_id: str
        :param text: Full text to chunk.
        :type text: str
        :param starting_chunk_id: Starting chunk identifier.
        :type starting_chunk_id: int
        :return: Chunk list.
        :rtype: list[TextChunk]
        """
        raise NotImplementedError


class FixedCharWindowChunker(Chunker):
    """
    Chunker that produces overlapping fixed-size character windows.
    """

    chunker_id = "fixed-char-window"

    def __init__(self, *, window_characters: int, overlap_characters: int) -> None:
        self._window_characters = int(window_characters)
        self._overlap_characters = int(overlap_characters)
        if self._window_characters <= 0:
            raise ValueError("window_characters must be greater than 0")
        if self._overlap_characters < 0:
            raise ValueError("overlap_characters must be greater than or equal to 0")
        if self._overlap_characters >= self._window_characters:
            raise ValueError("overlap_characters must be less than window_characters")

    def chunk_text(self, *, item_id: str, text: str, starting_chunk_id: int) -> List[TextChunk]:
        """
        Chunk a text string into fixed-size character windows.

        :param item_id: Item identifier.
        :type item_id: str
        :param text: Input text.
        :type text: str
        :param starting_chunk_id: Starting chunk identifier.
        :type starting_chunk_id: int
        :return: Chunk list.
        :rtype: list[TextChunk]
        """
        chunks: List[TextChunk] = []
        chunk_id = int(starting_chunk_id)
        position = 0
        stride = self._window_characters - self._overlap_characters
        while position < len(text):
            span_start = position
            span_end = min(position + self._window_characters, len(text))
            chunk_text = text[span_start:span_end]
            if chunk_text.strip():
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        item_id=item_id,
                        span_start=span_start,
                        span_end=span_end,
                        text=chunk_text,
                    )
                )
                chunk_id += 1
            if span_end >= len(text):
                position = len(text)
            else:
                position += stride
        return chunks


class ParagraphChunker(Chunker):
    """
    Chunker that produces paragraph spans separated by blank lines.
    """

    chunker_id = "paragraph"

    def chunk_text(self, *, item_id: str, text: str, starting_chunk_id: int) -> List[TextChunk]:
        """
        Chunk a text string by paragraph boundaries.

        :param item_id: Item identifier.
        :type item_id: str
        :param text: Input text.
        :type text: str
        :param starting_chunk_id: Starting chunk identifier.
        :type starting_chunk_id: int
        :return: Chunk list.
        :rtype: list[TextChunk]
        """
        import re

        chunks: List[TextChunk] = []
        chunk_id = int(starting_chunk_id)
        for match in re.finditer(r"(?:[^\n]|\n(?!\n))+", text):
            span_start = int(match.start())
            span_end = int(match.end())
            chunk_text = text[span_start:span_end]
            if not chunk_text.strip():
                continue
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    item_id=item_id,
                    span_start=span_start,
                    span_end=span_end,
                    text=chunk_text,
                )
            )
            chunk_id += 1
        return chunks


class FixedTokenWindowChunker(Chunker):
    """
    Chunker that produces overlapping fixed-size token windows.
    """

    chunker_id = "fixed-token-window"

    def __init__(self, *, window_tokens: int, overlap_tokens: int, tokenizer: Tokenizer) -> None:
        self._window_tokens = int(window_tokens)
        self._overlap_tokens = int(overlap_tokens)
        self._tokenizer = tokenizer
        if self._window_tokens <= 0:
            raise ValueError("window_tokens must be greater than 0")
        if self._overlap_tokens < 0:
            raise ValueError("overlap_tokens must be greater than or equal to 0")
        if self._overlap_tokens >= self._window_tokens:
            raise ValueError("overlap_tokens must be less than window_tokens")

    def chunk_text(self, *, item_id: str, text: str, starting_chunk_id: int) -> List[TextChunk]:
        """
        Chunk a text string into fixed-size token windows.

        :param item_id: Item identifier.
        :type item_id: str
        :param text: Input text.
        :type text: str
        :param starting_chunk_id: Starting chunk identifier.
        :type starting_chunk_id: int
        :return: Chunk list.
        :rtype: list[TextChunk]
        """
        token_spans = self._tokenizer.tokenize(text)
        if not token_spans:
            return []
        chunks: List[TextChunk] = []
        chunk_id = int(starting_chunk_id)
        stride = self._window_tokens - self._overlap_tokens
        token_index = 0
        while token_index < len(token_spans):
            window_end = min(token_index + self._window_tokens, len(token_spans))
            span_start = token_spans[token_index].span_start
            span_end = token_spans[window_end - 1].span_end
            chunk_text = text[span_start:span_end]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    item_id=item_id,
                    span_start=span_start,
                    span_end=span_end,
                    text=chunk_text,
                )
            )
            chunk_id += 1
            if window_end >= len(token_spans):
                token_index = len(token_spans)
            else:
                token_index += stride
        return chunks


class TokenizerConfig(BaseModel):
    """
    Configuration for tokenizer selection.

    :ivar tokenizer_id: Tokenizer identifier.
    :vartype tokenizer_id: str
    """

    model_config = ConfigDict(extra="forbid")

    tokenizer_id: str = Field(min_length=1)

    def build_tokenizer(self) -> Tokenizer:
        """
        Build a tokenizer instance from this configuration.

        :return: Tokenizer instance.
        :rtype: Tokenizer
        :raises ValueError: If the tokenizer identifier is unknown.
        """
        if self.tokenizer_id == WhitespaceTokenizer.tokenizer_id:
            return WhitespaceTokenizer()
        raise ValueError(f"Unknown tokenizer_id: {self.tokenizer_id!r}")


class ChunkerConfig(BaseModel):
    """
    Configuration for chunker selection.

    :ivar chunker_id: Chunker identifier.
    :vartype chunker_id: str
    :ivar window_characters: Window size for fixed character chunking.
    :vartype window_characters: int or None
    :ivar overlap_characters: Overlap size for fixed character chunking.
    :vartype overlap_characters: int or None
    :ivar window_tokens: Window size for fixed token chunking.
    :vartype window_tokens: int or None
    :ivar overlap_tokens: Overlap size for fixed token chunking.
    :vartype overlap_tokens: int or None
    """

    model_config = ConfigDict(extra="forbid")

    chunker_id: str = Field(min_length=1)
    window_characters: Optional[int] = Field(default=None, ge=1)
    overlap_characters: Optional[int] = Field(default=None, ge=0)
    window_tokens: Optional[int] = Field(default=None, ge=1)
    overlap_tokens: Optional[int] = Field(default=None, ge=0)

    def build_chunker(self, *, tokenizer: Optional[Tokenizer]) -> Chunker:
        """
        Build a chunker instance from this configuration.

        :param tokenizer: Tokenizer used by token-window chunking strategies.
        :type tokenizer: Tokenizer or None
        :return: Chunker instance.
        :rtype: Chunker
        :raises ValueError: If required configuration is missing or unknown.
        """
        if self.chunker_id == FixedCharWindowChunker.chunker_id:
            if self.window_characters is None or self.overlap_characters is None:
                raise ValueError(
                    "fixed-char-window requires window_characters and overlap_characters"
                )
            return FixedCharWindowChunker(
                window_characters=self.window_characters,
                overlap_characters=self.overlap_characters,
            )
        if self.chunker_id == ParagraphChunker.chunker_id:
            return ParagraphChunker()
        if self.chunker_id == FixedTokenWindowChunker.chunker_id:
            if self.window_tokens is None or self.overlap_tokens is None:
                raise ValueError("fixed-token-window requires window_tokens and overlap_tokens")
            if tokenizer is None:
                raise ValueError("tokenizer configuration is required for fixed-token-window")
            return FixedTokenWindowChunker(
                window_tokens=self.window_tokens,
                overlap_tokens=self.overlap_tokens,
                tokenizer=tokenizer,
            )
        raise ValueError(f"Unknown chunker_id: {self.chunker_id!r}")
