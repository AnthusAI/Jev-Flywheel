"""
Agentic text linking using virtual file edits.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from jinja2 import Environment, StrictUndefined

from .markup import (
    TextAnnotatedSpan,
    build_span_context_section,
    parse_span_markup,
    strip_span_tags,
)
from .models import TextLinkRequest, TextLinkResult
from .tool_loop import request_confirmation, run_tool_loop

DEFAULT_LINK_ID_PREFIX = "link_"


def apply_text_link(request: TextLinkRequest) -> TextLinkResult:
    """
    Apply text linking using a language model.

    :param request: Text link request.
    :type request: TextLinkRequest
    :return: Text link result.
    :rtype: TextLinkResult
    :raises ValueError: If model output is invalid or text is modified. Empty outputs trigger
        a confirmation round and return a warning when confirmed.
    """
    warnings: List[str] = []
    system_prompt = _render_system_prompt(
        request.system_prompt,
        id_prefix=request.id_prefix,
    )

    if request.mock_marked_up_text is not None:
        return _build_mock_result(
            request,
            request.mock_marked_up_text,
        )

    result = run_tool_loop(
        text=request.text,
        client=request.client,
        system_prompt=system_prompt,
        prompt_template=request.prompt_template,
        max_rounds=request.max_rounds,
        max_edits_per_round=request.max_edits_per_round,
        apply_str_replace=_apply_link_replace,
        validate_text=lambda current_text: _validate_link_markup(current_text, request.id_prefix),
        build_retry_message=lambda errors, current_text: _build_retry_message(
            errors, current_text, request.id_prefix
        ),
    )

    if not result.done:
        if result.last_error:
            recovered = _attempt_missing_coverage_recovery(
                marked_up_text=result.text,
                id_prefix=request.id_prefix,
                warnings=warnings,
            )
            if recovered is not None:
                return recovered
            raise ValueError(f"Text link failed: {result.last_error}")
        warnings.append("Text link reached max rounds without done=true")

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
            apply_str_replace=_apply_link_replace,
            confirmation_message=_build_empty_confirmation_message(result.text),
            validate_text=lambda current_text: _validate_link_markup(
                current_text, request.id_prefix
            ),
            build_retry_message=lambda errors, current_text: _build_retry_message(
                errors, current_text, request.id_prefix
            ),
        )
        if not confirmation.done:
            if confirmation.last_error:
                raise ValueError(f"Text link failed: {confirmation.last_error}")
            warnings.append("Text link confirmation reached max rounds without done=true")
        _validate_preserved_text(original=request.text, marked_up=confirmation.text)
        if confirmation.text == request.text:
            warnings.append("Text link returned no spans; model confirmed empty result")
            return TextLinkResult(marked_up_text=confirmation.text, spans=[], warnings=warnings)
        spans = parse_span_markup(confirmation.text)
        errors = _validate_link_spans(spans, request.id_prefix)
        if errors:
            raise ValueError("; ".join(errors))
        return TextLinkResult(marked_up_text=confirmation.text, spans=spans, warnings=warnings)

    _validate_preserved_text(original=request.text, marked_up=result.text)
    spans = parse_span_markup(result.text)
    errors = _validate_link_spans(spans, request.id_prefix)
    if errors:
        autofilled = _autofill_ref_spans(result.text, spans)
        if autofilled is not None:
            result_text, spans, autofill_warnings = autofilled
            warnings.extend(autofill_warnings)
            errors = _validate_link_spans(spans, request.id_prefix)
            if errors:
                raise ValueError("; ".join(errors))
            return TextLinkResult(
                marked_up_text=result_text,
                spans=spans,
                warnings=warnings,
            )
        raise ValueError("; ".join(errors))

    return TextLinkResult(marked_up_text=result.text, spans=spans, warnings=warnings)


def _build_mock_result(
    request: TextLinkRequest,
    marked_up_text: str,
) -> TextLinkResult:
    if marked_up_text == request.text:
        raise ValueError("Text link produced no spans")
    _validate_preserved_text(original=request.text, marked_up=marked_up_text)
    spans = parse_span_markup(marked_up_text)
    errors = _validate_link_spans(spans, request.id_prefix)
    if errors:
        raise ValueError("; ".join(errors))
    return TextLinkResult(marked_up_text=marked_up_text, spans=spans, warnings=[])


def _render_system_prompt(template: str, *, id_prefix: str) -> str:
    env = Environment(undefined=StrictUndefined, autoescape=True)
    rendered = env.from_string(template).render(
        id_prefix=id_prefix,
    )
    return rendered


def _apply_link_replace(text: str, old_str: str, new_str: str) -> str:
    occurrences = text.count(old_str)
    if occurrences == 0:
        raise ValueError("Text link replacement old_str not found")
    if occurrences > 1:
        raise ValueError("Text link replacement old_str is not unique")
    _validate_replace_text(old_str, new_str)
    return text.replace(old_str, new_str, 1)


def _validate_replace_text(old_str: str, new_str: str) -> None:
    if "<span" in old_str or "</span>" in old_str:
        raise ValueError("Text link replacements must target plain text without span tags")
    if strip_span_tags(old_str) != strip_span_tags(new_str):
        raise ValueError("Text link replacements may only insert span tags")


def _validate_preserved_text(*, original: str, marked_up: str) -> None:
    if strip_span_tags(marked_up) != original:
        raise ValueError("Text link edits modified the source text")


def _validate_link_markup(marked_up_text: str, id_prefix: str) -> List[str]:
    try:
        spans = parse_span_markup(marked_up_text)
    except ValueError as exc:
        return [str(exc)]
    errors = _validate_link_spans(spans, id_prefix)
    errors.extend(_validate_link_coverage(marked_up_text, spans))
    errors.extend(_validate_link_span_minimality(spans))
    return errors


def _attempt_missing_coverage_recovery(
    *,
    marked_up_text: str,
    id_prefix: str,
    warnings: List[str],
) -> Optional[TextLinkResult]:
    try:
        spans = parse_span_markup(marked_up_text)
    except ValueError:
        return None
    plain_text = strip_span_tags(marked_up_text)
    promoted_spans, promotion_warnings = _promote_ref_spans_to_id(
        spans=spans,
        id_prefix=id_prefix,
    )
    if promotion_warnings:
        warnings.extend(promotion_warnings)
        spans = promoted_spans
        marked_up_text = _render_span_markup(plain_text, spans)
    errors = _validate_link_spans(spans, id_prefix)
    errors.extend(_validate_link_coverage(marked_up_text, spans))
    errors.extend(_validate_link_span_minimality(spans))
    if not errors or not _errors_are_missing_coverage_only(errors):
        return None
    autofilled = _autofill_ref_spans(marked_up_text, spans)
    if autofilled is None:
        return None
    result_text, result_spans, autofill_warnings = autofilled
    warnings.extend(autofill_warnings)
    errors_after = _validate_link_spans(result_spans, id_prefix)
    errors_after.extend(_validate_link_coverage(result_text, result_spans))
    errors_after.extend(_validate_link_span_minimality(result_spans))
    if errors_after:
        return None
    return TextLinkResult(
        marked_up_text=result_text,
        spans=result_spans,
        warnings=warnings,
    )


def _errors_are_missing_coverage_only(errors: Sequence[str]) -> bool:
    return all(_is_ref_coverage_error(error) for error in errors)


def _is_ref_coverage_error(error: str) -> bool:
    if error.startswith("Missing linked spans for repeated text"):
        return True
    if error.startswith("Id '") and error.endswith("must have at least one ref span"):
        return True
    if error.startswith("Repeated text '") and error.endswith("must include ref spans for repeats"):
        return True
    return False


def _promote_ref_spans_to_id(
    *,
    spans: Sequence[TextAnnotatedSpan],
    id_prefix: str,
) -> Tuple[List[TextAnnotatedSpan], List[str]]:
    spans_by_text: Dict[str, List[TextAnnotatedSpan]] = {}
    for span in spans:
        spans_by_text.setdefault(span.text, []).append(span)
    promote_indices: Dict[int, str] = {}
    warnings: List[str] = []
    for span_text, group in spans_by_text.items():
        if any("id" in span.attributes for span in group):
            continue
        ref_spans = [span for span in group if "ref" in span.attributes]
        if not ref_spans:
            continue
        candidate = sorted(ref_spans, key=lambda span: (span.start_char, span.end_char))[0]
        ref_value = candidate.attributes.get("ref", "")
        if not ref_value.startswith(id_prefix):
            continue
        promote_indices[candidate.index] = ref_value
        warnings.append(f"Promoted ref span for '{span_text}' to id '{ref_value}'.")
    if not promote_indices:
        return list(spans), []
    updated: List[TextAnnotatedSpan] = []
    for span in spans:
        attributes = span.attributes
        if span.index in promote_indices:
            attributes = {"id": promote_indices[span.index]}
        updated.append(
            TextAnnotatedSpan(
                index=span.index,
                start_char=span.start_char,
                end_char=span.end_char,
                text=span.text,
                attributes=attributes,
            )
        )
    return updated, warnings


def _validate_link_spans(spans: Iterable[TextAnnotatedSpan], id_prefix: str) -> List[str]:
    errors: List[str] = []
    seen_ids: List[str] = []
    id_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    span_texts_by_id: Dict[str, Dict[str, set[str]]] = {}
    spans_by_text: Dict[str, List[TextAnnotatedSpan]] = {}
    for span in spans:
        if len(span.attributes) != 1:
            errors.append(f"Span {span.index} must include exactly one attribute")
            continue
        name, value = next(iter(span.attributes.items()))
        if value.strip() == "":
            errors.append(f"Span {span.index} has an empty value for attribute '{name}'")
            continue
        if name == "id":
            if not value.startswith(id_prefix):
                errors.append(f"Span {span.index} id '{value}' must start with '{id_prefix}'")
            if value in seen_ids:
                errors.append(f"Span {span.index} uses duplicate id '{value}'")
            seen_ids.append(value)
            id_counts[value] = id_counts.get(value, 0) + 1
            span_texts_by_id.setdefault(value, {"id": set(), "ref": set()})["id"].add(span.text)
        elif name == "ref":
            if value not in seen_ids:
                errors.append(f"Span {span.index} ref '{value}' does not match a previous id")
            ref_counts[value] = ref_counts.get(value, 0) + 1
            span_texts_by_id.setdefault(value, {"id": set(), "ref": set()})["ref"].add(span.text)
        else:
            errors.append(
                f"Span {span.index} uses attribute '{name}' but only 'id' or 'ref' are allowed"
            )
        spans_by_text.setdefault(span.text, []).append(span)
    for id_value, text_sets in span_texts_by_id.items():
        id_texts = text_sets["id"]
        ref_texts = text_sets["ref"]
        if len(id_texts) > 1:
            errors.append(f"Id '{id_value}' spans must wrap the same text")
        if len(ref_texts) > 1:
            errors.append(f"Ref spans for id '{id_value}' must wrap the same text")
        if id_texts and ref_texts and id_texts != ref_texts:
            id_text = next(iter(id_texts))
            ref_text = next(iter(ref_texts))
            errors.append(
                f"Id '{id_value}' span text must match ref span text "
                f"(id: '{id_text}', ref: '{ref_text}')"
            )
    for span_text, span_group in spans_by_text.items():
        if len(span_group) <= 1:
            continue
        id_values = [span.attributes.get("id") for span in span_group if "id" in span.attributes]
        ref_values = [span.attributes.get("ref") for span in span_group if "ref" in span.attributes]
        if len(id_values) != 1:
            errors.append(f"Repeated text '{span_text}' must have exactly one id span")
            continue
        id_value = id_values[0]
        if not ref_values:
            errors.append(f"Repeated text '{span_text}' must include ref spans for repeats")
            continue
        if any(ref != id_value for ref in ref_values):
            errors.append(f"Repeated text '{span_text}' refs must match id '{id_value}'")
    for id_value, count in id_counts.items():
        if count > 1:
            continue
        if ref_counts.get(id_value, 0) == 0:
            errors.append(f"Id '{id_value}' must have at least one ref span")
    return errors


def _validate_link_coverage(marked_up_text: str, spans: Iterable[TextAnnotatedSpan]) -> List[str]:
    plain_text = strip_span_tags(marked_up_text)
    spans_by_text: Dict[str, int] = {}
    for span in spans:
        if span.text:
            spans_by_text[span.text] = spans_by_text.get(span.text, 0) + 1
    errors: List[str] = []
    for span_text, span_count in spans_by_text.items():
        occurrences = plain_text.count(span_text)
        if occurrences > span_count:
            errors.append(
                f"Missing linked spans for repeated text '{span_text}' "
                f"({span_count}/{occurrences})"
            )
    return errors


_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _validate_link_span_minimality(spans: Iterable[TextAnnotatedSpan]) -> List[str]:
    errors: List[str] = []
    for span in spans:
        tokens = [token for token in _WORD_PATTERN.findall(span.text) if token]
        if not tokens:
            continue
        counts = Counter(tokens)
        repeated = [token for token, count in counts.items() if count > 1]
        if repeated:
            errors.append(
                f"Span {span.index} contains repeated token '{repeated[0]}'. "
                "Split repeated mentions into separate spans."
            )
    return errors


def _autofill_ref_spans(
    marked_up_text: str,
    spans: Iterable[TextAnnotatedSpan],
) -> Optional[Tuple[str, List[TextAnnotatedSpan], List[str]]]:
    plain_text = strip_span_tags(marked_up_text)
    existing_spans = list(spans)
    occupied = [(span.start_char, span.end_char) for span in existing_spans]
    new_spans: List[TextAnnotatedSpan] = []

    def is_covered(start: int, end: int) -> bool:
        return any(start < span_end and end > span_start for span_start, span_end in occupied)

    for span in existing_spans:
        id_value = span.attributes.get("id")
        if not id_value:
            continue
        span_text = span.text
        if not span_text:
            continue
        for match in re.finditer(re.escape(span_text), plain_text):
            start = match.start()
            end = match.end()
            if is_covered(start, end):
                continue
            ref_span = TextAnnotatedSpan(
                index=1,
                start_char=start,
                end_char=end,
                text=span_text,
                attributes={"ref": id_value},
            )
            new_spans.append(ref_span)
            occupied.append((start, end))

    if not new_spans:
        return None

    merged_spans = sorted(
        existing_spans + new_spans,
        key=lambda span: (span.start_char, span.end_char),
    )
    reindexed: List[TextAnnotatedSpan] = []
    for index, span in enumerate(merged_spans, start=1):
        reindexed.append(
            TextAnnotatedSpan(
                index=index,
                start_char=span.start_char,
                end_char=span.end_char,
                text=span.text,
                attributes=span.attributes,
            )
        )
    rendered = _render_span_markup(plain_text, reindexed)
    warnings = [f"Autofilled {len(new_spans)} ref spans for repeated text."]
    return rendered, reindexed, warnings


def _render_span_markup(text: str, spans: List[TextAnnotatedSpan]) -> str:
    parts: List[str] = []
    cursor = 0
    for span in spans:
        if span.start_char < cursor:
            raise ValueError("Span overlap detected while rendering markup")
        parts.append(text[cursor : span.start_char])
        attrs = " ".join(f'{key}="{value}"' for key, value in span.attributes.items())
        if attrs:
            parts.append(f"<span {attrs}>")
        else:
            parts.append("<span>")
        parts.append(text[span.start_char : span.end_char])
        parts.append("</span>")
        cursor = span.end_char
    parts.append(text[cursor:])
    return "".join(parts)


def _build_retry_message(errors: Sequence[str], current_text: str, id_prefix: str) -> str:
    error_lines = "\n".join(f"- {error}" for error in errors)
    context_section = build_span_context_section(current_text, errors)
    coverage_guidance = _build_coverage_guidance(errors)
    nested_guidance = ""
    if any("nested span" in error for error in errors):
        nested_guidance = "Do not create nested or overlapping spans. Remove nested spans and wrap only bare text.\n"
    return (
        "Your last edit did not validate.\n"
        "Issues:\n"
        f"{error_lines}\n\n"
        f"{context_section}"
        f"{coverage_guidance}"
        f"{nested_guidance}"
        "Please fix the markup using str_replace. Use id for first mentions and ref for repeats. "
        "Reuse the same id for identical names and do not assign multiple ids to the same name. "
        f"Ids must start with '{id_prefix}'. Try again.\n"
        "Current text:\n"
        f"---\n{current_text}\n---"
    )


def _build_coverage_guidance(errors: Sequence[str]) -> str:
    instructions: List[str] = []
    for error in errors:
        match = re.match(
            r"Missing linked spans for repeated text '(.+)' \((\d+)/(\d+)\)",
            error,
        )
        if match:
            span_text = match.group(1)
            instructions.append(
                f"- Add ref spans for every remaining occurrence of '{span_text}' "
                "using the same id as its first mention."
            )
            continue
        if error.startswith("Id '") and error.endswith("must have at least one ref span"):
            id_value = error.split("'")[1]
            instructions.append(f'- Add ref spans with ref="{id_value}" for each later occurrence.')
            continue
        if error.startswith("Repeated text '") and error.endswith(
            "must include ref spans for repeats"
        ):
            span_text = error.split("'")[1]
            instructions.append(
                f"- Ensure '{span_text}' has one id on the first mention and ref spans on later mentions."
            )
            continue
        if error.startswith("Id '") and "span text must match ref span text" in error:
            id_value = error.split("'")[1]
            instructions.append(
                f"- Ensure every span with id/ref '{id_value}' wraps the exact same text."
            )
    if not instructions:
        return ""
    return "Fixes:\n" + "\n".join(instructions) + "\n\n"


def _build_empty_confirmation_message(text: str) -> str:
    return (
        "No linked spans were inserted. If there are truly no repeated names to link, "
        "call done again without changes. Otherwise insert id/ref spans for the repeated names.\n"
        "Current text:\n"
        f"---\n{text}\n---"
    )
