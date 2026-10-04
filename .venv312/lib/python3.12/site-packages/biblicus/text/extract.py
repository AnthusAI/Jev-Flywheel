"""
Agentic text extraction using virtual file edits.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .models import TextExtractRequest, TextExtractResult, TextExtractSpan
from .tool_loop import request_confirmation, run_tool_loop


def apply_text_extract(request: TextExtractRequest) -> TextExtractResult:
    """
    Apply text extraction using a language model.

    :param request: Text extract request.
    :type request: TextExtractRequest
    :return: Text extract result.
    :rtype: TextExtractResult
    :raises ValueError: If model output is invalid or text is modified. Empty outputs trigger
        a confirmation round and return a warning when confirmed.
    """
    if request.mock_marked_up_text is not None:
        return _build_mock_result(request, request.mock_marked_up_text)

    warnings: List[str] = []
    result = run_tool_loop(
        text=request.text,
        client=request.client,
        system_prompt=request.system_prompt,
        prompt_template=request.prompt_template,
        max_rounds=request.max_rounds,
        max_edits_per_round=request.max_edits_per_round,
        apply_str_replace=_apply_extract_replace,
        validate_text=_validate_extract_markup,
        build_retry_message=_build_retry_message,
    )
    if not result.done:
        if result.last_error:
            message_error = _extract_validation_error_from_messages(result.messages)
            if message_error:
                if request.normalize_nested_spans and "nested spans" in message_error:
                    normalized = _normalize_nested_spans(result.text)
                    _validate_preserved_text(original=request.text, marked_up=normalized)
                    spans = _extract_spans(marked_up_text=normalized)
                    warnings.append("Text extract normalized nested spans from model output")
                    return TextExtractResult(
                        marked_up_text=normalized,
                        spans=spans,
                        warnings=warnings,
                    )
                raise ValueError(f"Text extract failed: {message_error}")
            raise ValueError(f"Text extract failed: {result.last_error}")
        warnings.append("Text extract reached max rounds without done=true")
    if result.text == request.text:
        if result.last_error:
            raise ValueError(result.last_error)
        confirmation = request_confirmation(
            result=result,
            text=result.text,
            client=request.client,
            system_prompt=request.system_prompt,
            prompt_template=request.prompt_template,
            max_rounds=2,
            max_edits_per_round=request.max_edits_per_round,
            apply_str_replace=_apply_extract_replace,
            confirmation_message=_build_empty_confirmation_message(result.text),
        )
        if not confirmation.done:
            if confirmation.last_error:
                raise ValueError(f"Text extract failed: {confirmation.last_error}")
            warnings.append("Text extract confirmation reached max rounds without done=true")
        _validate_preserved_text(original=request.text, marked_up=confirmation.text)
        spans = _extract_spans(marked_up_text=confirmation.text)
        if not spans:
            warnings.append("Text extract returned no spans; model confirmed empty result")
        return TextExtractResult(
            marked_up_text=confirmation.text,
            spans=spans,
            warnings=warnings,
        )
    _validate_preserved_text(original=request.text, marked_up=result.text)
    spans = _extract_spans(marked_up_text=result.text)
    return TextExtractResult(marked_up_text=result.text, spans=spans, warnings=warnings)


def _build_mock_result(request: TextExtractRequest, marked_up_text: str) -> TextExtractResult:
    if marked_up_text == request.text:
        raise ValueError("Text extract produced no spans")
    _validate_preserved_text(original=request.text, marked_up=marked_up_text)
    spans = _extract_spans(marked_up_text=marked_up_text)
    return TextExtractResult(marked_up_text=marked_up_text, spans=spans, warnings=[])


def _apply_extract_replace(text: str, old_str: str, new_str: str) -> str:
    occurrences = text.count(old_str)
    if occurrences == 0:
        raise ValueError("Text extract replacement old_str not found")
    if occurrences > 1:
        raise ValueError("Text extract replacement old_str is not unique")
    _validate_replace_text(old_str, new_str)
    return text.replace(old_str, new_str, 1)


def _validate_replace_text(old_str: str, new_str: str) -> None:
    if _strip_span_tags(old_str) != _strip_span_tags(new_str):
        raise ValueError("Text extract replacements may only insert span tags")


def _validate_preserved_text(*, original: str, marked_up: str) -> None:
    if _strip_span_tags(marked_up) != original:
        raise ValueError("Text extract edits modified the source text")


def _strip_span_tags(text: str) -> str:
    return text.replace("<span>", "").replace("</span>", "")


def _normalize_nested_spans(marked_up_text: str) -> str:
    open_tag = "<span>"
    close_tag = "</span>"
    tag_pattern = re.compile(re.escape(open_tag) + "|" + re.escape(close_tag))
    output: List[str] = []
    cursor = 0
    depth = 0

    for match in tag_pattern.finditer(marked_up_text):
        output.append(marked_up_text[cursor : match.start()])
        tag = match.group(0)
        if tag == open_tag:
            if depth == 0:
                output.append(tag)
            depth += 1
        else:
            if depth > 0:
                if depth == 1:
                    output.append(tag)
                depth -= 1
            else:
                output.append(tag)
        cursor = match.end()

    output.append(marked_up_text[cursor:])
    return "".join(output)


def _extract_spans(*, marked_up_text: str) -> List[TextExtractSpan]:
    open_tag = "<span>"
    close_tag = "</span>"
    tag_pattern = re.compile(re.escape(open_tag) + "|" + re.escape(close_tag))
    spans: List[TextExtractSpan] = []
    cursor = 0
    original_index = 0
    span_start = None
    span_text = ""

    for match in tag_pattern.finditer(marked_up_text):
        chunk = marked_up_text[cursor : match.start()]
        if chunk:
            if span_start is not None:
                span_text += chunk
            original_index += len(chunk)
        tag = match.group(0)
        if tag == open_tag:
            if span_start is not None:
                raise ValueError("Text extract contains nested spans")
            span_start = original_index
            span_text = ""
        else:
            if span_start is None:
                raise ValueError("Text extract contains an unmatched closing tag")
            span_end = original_index
            spans.append(
                TextExtractSpan(
                    index=len(spans) + 1,
                    start_char=span_start,
                    end_char=span_end,
                    text=span_text,
                )
            )
            span_start = None
            span_text = ""
        cursor = match.end()

    tail = marked_up_text[cursor:]
    if tail:
        if span_start is not None:
            span_text += tail
        original_index += len(tail)

    if span_start is not None:
        raise ValueError("Text extract contains an unclosed span")

    return spans


def _validate_extract_markup(marked_up_text: str) -> List[str]:
    try:
        _extract_spans(marked_up_text=marked_up_text)
    except ValueError as exc:
        return [str(exc)]
    return []


def _build_retry_message(errors: List[str], current_text: str) -> str:
    error_lines = "\n".join(f"- {error}" for error in errors)
    return (
        "Your last edit did not validate.\n"
        "Issues:\n"
        f"{error_lines}\n\n"
        "Please fix the markup using str_replace. "
        "Do not nest <span> tags and do not create unmatched tags.\n"
        "Current text:\n"
        f"---\n{current_text}\n---"
    )


def _extract_validation_error_from_messages(
    messages: List[dict[str, Any]],
) -> Optional[str]:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if "Your last edit did not validate." not in content:
            continue
        if "Issues:" not in content:
            continue
        lines = content.splitlines()
        try:
            issues_index = lines.index("Issues:")
        except ValueError:
            continue
        for line in lines[issues_index + 1 :]:
            if line.startswith("- "):
                return line[2:].strip()
    return None


def _build_empty_confirmation_message(text: str) -> str:
    return (
        "No spans were inserted. If there are truly no spans to return, call done again without changes. "
        "Otherwise insert <span> tags for the requested text.\n"
        "Current text:\n"
        f"---\n{text}\n---"
    )
