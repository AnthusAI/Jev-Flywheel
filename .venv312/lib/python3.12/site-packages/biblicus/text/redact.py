"""
Agentic text redaction using virtual file edits.
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
from .models import TextRedactRequest, TextRedactResult
from .tool_loop import request_confirmation, run_tool_loop

DEFAULT_REDACTION_TYPES = ["pii", "pci", "phi"]


def apply_text_redact(request: TextRedactRequest) -> TextRedactResult:
    """
    Apply text redaction using a language model.

    :param request: Text redact request.
    :type request: TextRedactRequest
    :return: Text redact result.
    :rtype: TextRedactResult
    :raises ValueError: If model output is invalid or text is modified. Empty outputs trigger
        a confirmation round and return a warning when confirmed.
    """
    warnings: List[str] = []
    redaction_types = _resolve_redaction_types(request.redaction_types)
    system_prompt = _render_system_prompt(
        request.system_prompt,
        redaction_types=redaction_types,
    )

    if request.mock_marked_up_text is not None:
        return _build_mock_result(
            request,
            request.mock_marked_up_text,
            redaction_types=redaction_types,
        )

    result = run_tool_loop(
        text=request.text,
        client=request.client,
        system_prompt=system_prompt,
        prompt_template=request.prompt_template,
        max_rounds=request.max_rounds,
        max_edits_per_round=request.max_edits_per_round,
        apply_str_replace=_apply_redact_replace,
        validate_text=lambda current_text: _validate_redaction_markup(
            current_text, redaction_types
        ),
        build_retry_message=lambda errors, current_text: _build_retry_message(
            errors, current_text, redaction_types
        ),
    )

    if not result.done:
        if result.last_error:
            raise ValueError(f"Text redact failed: {result.last_error}")
        warnings.append("Text redact reached max rounds without done=true")

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
            apply_str_replace=_apply_redact_replace,
            confirmation_message=_build_empty_confirmation_message(result.text),
            validate_text=lambda current_text: _validate_redaction_markup(
                current_text, redaction_types
            ),
            build_retry_message=lambda errors, current_text: _build_retry_message(
                errors, current_text, redaction_types
            ),
        )
        if not confirmation.done:
            if confirmation.last_error:
                raise ValueError(f"Text redact failed: {confirmation.last_error}")
            warnings.append("Text redact confirmation reached max rounds without done=true")
        _validate_preserved_text(original=request.text, marked_up=confirmation.text)
        spans = parse_span_markup(confirmation.text)
        validation_errors = _validate_redaction_spans(spans, redaction_types)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        if not spans:
            warnings.append("Text redact returned no spans; model confirmed empty result")
        return TextRedactResult(
            marked_up_text=confirmation.text,
            spans=spans,
            warnings=warnings,
        )

    _validate_preserved_text(original=request.text, marked_up=result.text)
    spans = parse_span_markup(result.text)
    validation_errors = _validate_redaction_spans(spans, redaction_types)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return TextRedactResult(marked_up_text=result.text, spans=spans, warnings=warnings)


def _build_mock_result(
    request: TextRedactRequest,
    marked_up_text: str,
    *,
    redaction_types: Sequence[str] | None,
) -> TextRedactResult:
    if marked_up_text == request.text:
        raise ValueError("Text redact produced no spans")
    _validate_preserved_text(original=request.text, marked_up=marked_up_text)
    spans = parse_span_markup(marked_up_text)
    errors = _validate_redaction_spans(spans, redaction_types)
    if errors:
        raise ValueError("; ".join(errors))
    return TextRedactResult(marked_up_text=marked_up_text, spans=spans, warnings=[])


def _resolve_redaction_types(redaction_types: Sequence[str] | None) -> List[str] | None:
    if redaction_types is None or len(redaction_types) == 0:
        return None
    return [value for value in redaction_types]


def _render_system_prompt(template: str, *, redaction_types: Sequence[str] | None) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=True)
    rendered = env.from_string(template).render(
        redaction_types=list(redaction_types) if redaction_types is not None else [],
    )
    return rendered


def _apply_redact_replace(text: str, old_str: str, new_str: str) -> str:
    occurrences = text.count(old_str)
    if occurrences == 0:
        raise ValueError("Text redact replacement old_str not found")
    if occurrences > 1:
        raise ValueError("Text redact replacement old_str is not unique")
    _validate_replace_text(old_str, new_str)
    return text.replace(old_str, new_str, 1)


def _validate_replace_text(old_str: str, new_str: str) -> None:
    if strip_span_tags(old_str) != strip_span_tags(new_str):
        raise ValueError("Text redact replacements may only insert span tags")


def _validate_preserved_text(*, original: str, marked_up: str) -> None:
    if strip_span_tags(marked_up) != original:
        raise ValueError("Text redact edits modified the source text")


def _validate_redaction_markup(
    marked_up_text: str, redaction_types: Sequence[str] | None
) -> List[str]:
    try:
        spans = parse_span_markup(marked_up_text)
    except ValueError as exc:
        return [str(exc)]
    return _validate_redaction_spans(spans, redaction_types)


def _validate_redaction_spans(
    spans: Iterable[TextAnnotatedSpan], redaction_types: Sequence[str] | None
) -> List[str]:
    errors: List[str] = []
    if redaction_types is None:
        for span in spans:
            if span.attributes:
                errors.append(
                    f"Span {span.index} contains attributes but redaction types are disabled"
                )
        return errors

    allowed_values = set(redaction_types)
    for span in spans:
        if len(span.attributes) != 1:
            errors.append(f"Span {span.index} must include exactly one redact attribute")
            continue
        name, value = next(iter(span.attributes.items()))
        if name != "redact":
            errors.append(f"Span {span.index} uses attribute '{name}' but only 'redact' is allowed")
        if value not in allowed_values:
            errors.append(
                f"Span {span.index} uses redaction type '{value}'. Allowed types: {', '.join(redaction_types)}"
            )
    return errors


def _build_retry_message(
    errors: Sequence[str], current_text: str, redaction_types: Sequence[str] | None
) -> str:
    error_lines = "\n".join(f"- {error}" for error in errors)
    context_section = build_span_context_section(current_text, errors)
    type_message = (
        "Do not add attributes."
        if redaction_types is None
        else f"Use a redact attribute with one of: {', '.join(redaction_types)}."
    )
    return (
        "Your last edit did not validate.\n"
        "Issues:\n"
        f"{error_lines}\n\n"
        f"{context_section}"
        "Please fix the markup using str_replace. "
        f"{type_message} Try again.\n"
        "Current text:\n"
        f"---\n{current_text}\n---"
    )


def _build_empty_confirmation_message(text: str) -> str:
    return (
        "No redaction spans were inserted. If there are truly no spans to return, "
        "call done again without changes. Otherwise insert the appropriate span tags.\n"
        "Current text:\n"
        f"---\n{text}\n---"
    )
