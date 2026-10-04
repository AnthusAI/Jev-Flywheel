"""
Pydantic models for agentic text utilities.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..ai.models import LlmClientConfig
from .markup import TextAnnotatedSpan
from .prompts import (
    DEFAULT_ANNOTATE_SYSTEM_PROMPT,
    DEFAULT_EXTRACT_SYSTEM_PROMPT,
    DEFAULT_LINK_SYSTEM_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_REDACT_SYSTEM_PROMPT,
    DEFAULT_SLICE_SYSTEM_PROMPT,
)


class TextToolLoopRequest(BaseModel):
    """
    Request to apply a tool-loop text operation using a language model.

    :param text: Input text to process.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt template containing ``{text}``. The base request requires
        callers to supply this. Specific utility requests provide built-in defaults so callers
        can typically omit it.
    :type system_prompt: str
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    :param mock_marked_up_text: Optional pre-rendered markup for deterministic tests.
    :type mock_marked_up_text: str or None
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    client: LlmClientConfig
    prompt_template: str = Field(default=DEFAULT_PROMPT_TEMPLATE, min_length=1)
    system_prompt: str = Field(min_length=1)
    max_rounds: int = Field(default=6, ge=1)
    max_edits_per_round: int = Field(default=500, ge=1)
    mock_marked_up_text: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_prompts(self) -> "TextToolLoopRequest":
        if "{text}" not in self.system_prompt:
            raise ValueError("system_prompt must include {text}")
        if "{text}" in self.prompt_template:
            raise ValueError("prompt_template must not include {text}")
        return self


class TextExtractRequest(TextToolLoopRequest):
    """
    Request to apply text extract using a language model.

    :param text: Input text to annotate with XML span tags.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt template containing ``{text}``. Defaults to the built-in
        text extract system prompt.
    :type system_prompt: str
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    :param normalize_nested_spans: Whether to normalize nested spans into a single span.
    :type normalize_nested_spans: bool
    """

    system_prompt: str = Field(default=DEFAULT_EXTRACT_SYSTEM_PROMPT, min_length=1)
    normalize_nested_spans: bool = Field(default=False)


class TextSliceRequest(TextToolLoopRequest):
    """
    Request to apply text slice using a language model.

    :param text: Input text to mark with slice markers.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt template containing ``{text}``. Defaults to the built-in
        text slice system prompt.
    :type system_prompt: str
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    """

    system_prompt: str = Field(default=DEFAULT_SLICE_SYSTEM_PROMPT, min_length=1)


class TextAnnotateRequest(TextToolLoopRequest):
    """
    Request to apply text annotation using span attributes.

    :param text: Input text to annotate with XML span tags.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt containing ``{text}``. Defaults to the built-in
        text annotate system prompt.
    :type system_prompt: str
    :param allowed_attributes: Optional list of allowed span attribute names.
    :type allowed_attributes: list[str] or None
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    """

    system_prompt: str = Field(default=DEFAULT_ANNOTATE_SYSTEM_PROMPT, min_length=1)
    allowed_attributes: Optional[List[str]] = None


class TextRedactRequest(TextToolLoopRequest):
    """
    Request to apply text redaction using span markers.

    :param text: Input text to annotate with XML span tags.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt containing ``{text}``. Defaults to the built-in
        text redact system prompt.
    :type system_prompt: str
    :param redaction_types: Optional list of allowed redaction types. When omitted, no attributes are allowed.
    :type redaction_types: list[str] or None
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    """

    system_prompt: str = Field(default=DEFAULT_REDACT_SYSTEM_PROMPT, min_length=1)
    redaction_types: Optional[List[str]] = None


class TextLinkRequest(TextToolLoopRequest):
    """
    Request to apply text linking using id/ref span attributes.

    :param text: Input text to annotate with XML span tags.
    :type text: str
    :param client: LLM client configuration.
    :type client: biblicus.ai.models.LlmClientConfig
    :param prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :type prompt_template: str
    :param system_prompt: System prompt containing ``{text}``. Defaults to the built-in
        text link system prompt.
    :type system_prompt: str
    :param id_prefix: Prefix required for id attributes.
    :type id_prefix: str
    :param max_rounds: Maximum number of edit rounds.
    :type max_rounds: int
    :param max_edits_per_round: Maximum edits per round.
    :type max_edits_per_round: int
    """

    system_prompt: str = Field(default=DEFAULT_LINK_SYSTEM_PROMPT, min_length=1)
    id_prefix: str = Field(default="link_", min_length=1)


class TextExtractSpan(BaseModel):
    """
    Extracted span of text.

    :param index: One-based index of the span in the output order.
    :type index: int
    :param start_char: Start character offset in the original text.
    :type start_char: int
    :param end_char: End character offset in the original text.
    :type end_char: int
    :param text: Span text.
    :type text: str
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str


class TextSliceSegment(BaseModel):
    """
    Extracted text slice.

    :param index: One-based index of the slice in the output order.
    :type index: int
    :param start_char: Start character offset in the original text.
    :type start_char: int
    :param end_char: End character offset in the original text.
    :type end_char: int
    :param text: Slice text.
    :type text: str
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str


class TextExtractResult(BaseModel):
    """
    Text extract output bundle.

    :param marked_up_text: Original text with XML span tags inserted.
    :type marked_up_text: str
    :param spans: Extracted spans in document order.
    :type spans: list[TextExtractSpan]
    :param warnings: Warning messages for the caller.
    :type warnings: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    marked_up_text: str
    spans: List[TextExtractSpan] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TextSliceResult(BaseModel):
    """
    Text slice output bundle.

    :param marked_up_text: Original text with slice markers inserted.
    :type marked_up_text: str
    :param slices: Extracted slices in document order.
    :type slices: list[TextSliceSegment]
    :param warnings: Warning messages for the caller.
    :type warnings: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    marked_up_text: str
    slices: List[TextSliceSegment] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TextAnnotateResult(BaseModel):
    """
    Text annotation output bundle.

    :param marked_up_text: Original text with XML span tags inserted.
    :type marked_up_text: str
    :param spans: Extracted spans in document order.
    :type spans: list[TextAnnotatedSpan]
    :param warnings: Warning messages for the caller.
    :type warnings: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    marked_up_text: str
    spans: List[TextAnnotatedSpan] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TextRedactResult(BaseModel):
    """
    Text redaction output bundle.

    :param marked_up_text: Original text with XML span tags inserted.
    :type marked_up_text: str
    :param spans: Redacted spans in document order.
    :type spans: list[TextAnnotatedSpan]
    :param warnings: Warning messages for the caller.
    :type warnings: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    marked_up_text: str
    spans: List[TextAnnotatedSpan] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class TextLinkResult(BaseModel):
    """
    Text linking output bundle.

    :param marked_up_text: Original text with XML span tags inserted.
    :type marked_up_text: str
    :param spans: Linked spans in document order.
    :type spans: list[TextAnnotatedSpan]
    :param warnings: Warning messages for the caller.
    :type warnings: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    marked_up_text: str
    spans: List[TextAnnotatedSpan] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
