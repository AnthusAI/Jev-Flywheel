"""
Shared span markup parsing utilities.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field


class TextAnnotatedSpan(BaseModel):
    """
    Span annotated with arbitrary attributes.

    :param index: One-based index of the span in the output order.
    :type index: int
    :param start_char: Start character offset in the original text.
    :type start_char: int
    :param end_char: End character offset in the original text.
    :type end_char: int
    :param text: Span text.
    :type text: str
    :param attributes: Attribute mapping extracted from the span tag.
    :type attributes: dict[str, str]
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str
    attributes: Dict[str, str] = Field(default_factory=dict)


_TAG_PATTERN = re.compile(r"<span\b[^>]*>|</span>")
_OPEN_TAG_PATTERN = re.compile(r"<span\b([^>]*)>")
_ATTRIBUTE_PATTERN = re.compile(r'([A-Za-z_][A-Za-z0-9_-]*)="([^"]*)"')
_SPAN_INDEX_PATTERN = re.compile(r"Span (\d+)")


def strip_span_tags(text: str) -> str:
    """
    Remove span tags from text.

    :param text: Text with span tags.
    :type text: str
    :return: Text with span tags removed.
    :rtype: str
    """
    return re.sub(r"</?span\b[^>]*>", "", text)


def parse_span_markup(marked_up_text: str) -> List[TextAnnotatedSpan]:
    """
    Parse span tags with attributes into annotated spans.

    :param marked_up_text: Text containing span tags.
    :type marked_up_text: str
    :return: Parsed spans with attributes.
    :rtype: list[TextAnnotatedSpan]
    :raises ValueError: If tags are malformed or nested.
    """
    spans: List[TextAnnotatedSpan] = []
    cursor = 0
    original_index = 0
    span_start = None
    span_text = ""
    span_attributes: Dict[str, str] = {}

    for match in _TAG_PATTERN.finditer(marked_up_text):
        chunk = marked_up_text[cursor : match.start()]
        if chunk:
            if span_start is not None:
                span_text += chunk
            original_index += len(chunk)
        tag = match.group(0)
        if tag.startswith("<span"):
            if span_start is not None:
                raise ValueError("Text markup contains nested spans")
            span_start = original_index
            span_text = ""
            span_attributes = _parse_span_attributes(tag)
        else:
            if span_start is None:
                raise ValueError("Text markup contains an unmatched closing tag")
            span_end = original_index
            spans.append(
                TextAnnotatedSpan(
                    index=len(spans) + 1,
                    start_char=span_start,
                    end_char=span_end,
                    text=span_text,
                    attributes=span_attributes,
                )
            )
            span_start = None
            span_text = ""
            span_attributes = {}
        cursor = match.end()

    tail = marked_up_text[cursor:]
    if tail:
        if span_start is not None:
            span_text += tail
        original_index += len(tail)

    if span_start is not None:
        raise ValueError("Text markup contains an unclosed span")

    return spans


def extract_span_indices(errors: Sequence[str]) -> List[int]:
    """
    Extract span indices referenced in error messages.

    :param errors: Validation error messages.
    :type errors: Sequence[str]
    :return: Sorted list of referenced span indices.
    :rtype: list[int]
    """
    indices: List[int] = []
    for error in errors:
        match = _SPAN_INDEX_PATTERN.search(error)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    return sorted(set(indices))


def summarize_span_context(marked_up_text: str, span_indices: Sequence[int]) -> List[str]:
    """
    Summarize span context for the requested indices.

    :param marked_up_text: Text containing span tags.
    :type marked_up_text: str
    :param span_indices: Span indices to summarize.
    :type span_indices: Sequence[int]
    :return: Human-readable span summaries.
    :rtype: list[str]
    :raises ValueError: If the markup is invalid.
    """
    spans = parse_span_markup(marked_up_text)
    span_by_index = {span.index: span for span in spans}
    summaries: List[str] = []
    for index in span_indices:
        span = span_by_index.get(index)
        if span is None:
            continue
        cleaned_text = " ".join(span.text.split())
        if cleaned_text:
            summaries.append(f"Span {index}: {cleaned_text}")
    return summaries


def build_span_context_section(marked_up_text: str, errors: Sequence[str]) -> str:
    """
    Build a formatted span context section for retry messages.

    :param marked_up_text: Text containing span tags.
    :type marked_up_text: str
    :param errors: Validation error messages.
    :type errors: Sequence[str]
    :return: Formatted span context block or empty string.
    :rtype: str
    """
    indices = extract_span_indices(errors)
    if not indices:
        return ""
    try:
        summaries = summarize_span_context(marked_up_text, indices)
    except ValueError:
        return ""
    if not summaries:
        return ""
    summary_lines = "\n".join(f"- {summary}" for summary in summaries)
    return f"Relevant spans:\n{summary_lines}\n\n"


def _parse_span_attributes(tag_text: str) -> Dict[str, str]:
    match = _OPEN_TAG_PATTERN.fullmatch(tag_text)
    if match is None:
        raise ValueError("Text markup contains an invalid span tag")
    attr_text = match.group(1).strip().replace('\\"', '"')
    if not attr_text:
        return {}
    attributes: Dict[str, str] = {}
    for attr_match in _ATTRIBUTE_PATTERN.finditer(attr_text):
        name = attr_match.group(1)
        value = attr_match.group(2)
        if name in attributes:
            raise ValueError("Text markup contains duplicate span attributes")
        attributes[name] = value
    cleaned = _ATTRIBUTE_PATTERN.sub("", attr_text).strip()
    if cleaned:
        raise ValueError("Text markup contains unsupported span attributes")
    return attributes
