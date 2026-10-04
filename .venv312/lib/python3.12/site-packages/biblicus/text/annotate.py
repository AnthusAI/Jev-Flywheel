"""
Agentic text annotation using virtual file edits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from jinja2 import Environment, StrictUndefined

from .markup import (
    TextAnnotatedSpan,
    build_span_context_section,
    parse_span_markup,
    strip_span_tags,
)
from .models import TextAnnotateRequest, TextAnnotateResult
from .tool_loop import request_confirmation, run_tool_loop

DEFAULT_ANNOTATION_ATTRIBUTES = ["label", "phase", "role", "evidence", "entity"]


def apply_text_annotate(request: TextAnnotateRequest) -> TextAnnotateResult:
    """
    Apply text annotation using a language model.

    :param request: Text annotate request.
    :type request: TextAnnotateRequest
    :return: Text annotate result.
    :rtype: TextAnnotateResult
    :raises ValueError: If model output is invalid or text is modified. Empty outputs trigger
        a confirmation round and return a warning when confirmed.
    """
    warnings: List[str] = []
    allowed_attributes = _resolve_allowed_attributes(request.allowed_attributes)
    system_prompt = _render_system_prompt(
        request.system_prompt,
        allowed_attributes=allowed_attributes,
    )

    if request.mock_marked_up_text is not None:
        return _build_mock_result(
            request,
            request.mock_marked_up_text,
            allowed_attributes=allowed_attributes,
        )

    result = run_tool_loop(
        text=request.text,
        client=request.client,
        system_prompt=system_prompt,
        prompt_template=request.prompt_template,
        max_rounds=request.max_rounds,
        max_edits_per_round=request.max_edits_per_round,
        apply_str_replace=_apply_annotate_replace,
        validate_text=lambda current_text: _validate_annotation_markup(
            current_text, allowed_attributes
        ),
        build_retry_message=lambda errors, current_text: _build_retry_message(
            errors, current_text, allowed_attributes
        ),
    )

    if not result.done:
        if result.last_error:
            raise ValueError(f"Text annotate failed: {result.last_error}")
        warnings.append("Text annotate reached max rounds without done=true")

    if result.text == request.text:
        if result.last_error:
            raise ValueError(result.last_error)
        confirmation = request_confirmation(
            result=result,
            text=result.text,
            client=request.client,
            system_prompt=system_prompt,
            prompt_template=request.prompt_template,
            max_rounds=2,
            max_edits_per_round=request.max_edits_per_round,
            apply_str_replace=_apply_annotate_replace,
            confirmation_message=_build_empty_confirmation_message(result.text),
            validate_text=lambda current_text: _validate_annotation_markup(
                current_text, allowed_attributes
            ),
            build_retry_message=lambda errors, current_text: _build_retry_message(
                errors, current_text, allowed_attributes
            ),
        )
        if not confirmation.done:
            if confirmation.last_error:
                raise ValueError(f"Text annotate failed: {confirmation.last_error}")
            warnings.append("Text annotate confirmation reached max rounds without done=true")
        _validate_preserved_text(original=request.text, marked_up=confirmation.text)
        spans = parse_span_markup(confirmation.text)
        validation_errors = _validate_annotation_spans(spans, allowed_attributes)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        if not spans:
            warnings.append("Text annotate returned no spans; model confirmed empty result")
        return TextAnnotateResult(
            marked_up_text=confirmation.text,
            spans=spans,
            warnings=warnings,
        )

    _validate_preserved_text(original=request.text, marked_up=result.text)
    spans = parse_span_markup(result.text)
    validation_errors = _validate_annotation_spans(spans, allowed_attributes)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return TextAnnotateResult(marked_up_text=result.text, spans=spans, warnings=warnings)


def _build_mock_result(
    request: TextAnnotateRequest,
    marked_up_text: str,
    *,
    allowed_attributes: Sequence[str],
) -> TextAnnotateResult:
    if marked_up_text == request.text:
        raise ValueError("Text annotate produced no spans")
    _validate_preserved_text(original=request.text, marked_up=marked_up_text)
    spans = parse_span_markup(marked_up_text)
    errors = _validate_annotation_spans(spans, allowed_attributes)
    if errors:
        raise ValueError("; ".join(errors))
    return TextAnnotateResult(marked_up_text=marked_up_text, spans=spans, warnings=[])


def _resolve_allowed_attributes(allowed: Sequence[str] | None) -> List[str]:
    if allowed is None:
        return list(DEFAULT_ANNOTATION_ATTRIBUTES)
    return [value for value in allowed]


def _render_system_prompt(template: str, *, allowed_attributes: Sequence[str]) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=True)
    rendered = env.from_string(template).render(
        allowed_attributes=list(allowed_attributes),
    )
    return rendered


def _apply_annotate_replace(text: str, old_str: str, new_str: str) -> str:
    occurrences = text.count(old_str)
    if occurrences == 0:
        raise ValueError("Text annotate replacement old_str not found")
    if occurrences > 1:
        raise ValueError("Text annotate replacement old_str is not unique")
    _validate_replace_text(old_str, new_str)
    return text.replace(old_str, new_str, 1)


def _validate_replace_text(old_str: str, new_str: str) -> None:
    if strip_span_tags(old_str) != strip_span_tags(new_str):
        raise ValueError("Text annotate replacements may only insert span tags")


def _validate_preserved_text(*, original: str, marked_up: str) -> None:
    if strip_span_tags(marked_up) != original:
        raise ValueError("Text annotate edits modified the source text")


def _validate_annotation_markup(
    marked_up_text: str, allowed_attributes: Sequence[str]
) -> List[str]:
    try:
        spans = parse_span_markup(marked_up_text)
    except ValueError as exc:
        return [str(exc)]
    return _validate_annotation_spans(spans, allowed_attributes)


def _validate_annotation_spans(
    spans: Iterable[TextAnnotatedSpan], allowed_attributes: Sequence[str]
) -> List[str]:
    errors: List[str] = []
    allowed_set = set(allowed_attributes)
    for span in spans:
        if not span.attributes:
            errors.append(
                f"Span {span.index} is missing an attribute. Allowed attributes: {', '.join(allowed_attributes)}"
            )
            continue
        if len(span.attributes) > 1:
            errors.append(f"Span {span.index} has multiple attributes; only one is allowed")
            continue
        name, value = next(iter(span.attributes.items()))
        if name not in allowed_set:
            errors.append(
                f"Span {span.index} uses attribute '{name}'. Allowed attributes: {', '.join(allowed_attributes)}"
            )
        if value.strip() == "":
            errors.append(f"Span {span.index} has an empty value for attribute '{name}'")
    return errors


def _build_retry_message(
    errors: Sequence[str], current_text: str, allowed_attributes: Sequence[str]
) -> str:
    error_lines = "\n".join(f"- {error}" for error in errors)
    context_section = build_span_context_section(current_text, errors)
    return (
        "Your last edit did not validate.\n"
        "Issues:\n"
        f"{error_lines}\n\n"
        f"{context_section}"
        "Please fix the markup using str_replace. Each span must include exactly one attribute. "
        "Allowed attributes are: "
        f"{', '.join(allowed_attributes)}. Try again.\n"
        "Current text:\n"
        f"---\n{current_text}\n---"
    )


def _build_empty_confirmation_message(text: str) -> str:
    return (
        "No annotated spans were inserted. If there are truly no spans to return, "
        "call done again without changes. Otherwise insert span tags with the correct attributes.\n"
        "Current text:\n"
        f"---\n{text}\n---"
    )
