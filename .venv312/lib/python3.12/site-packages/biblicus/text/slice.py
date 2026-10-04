"""
Agentic text slicing using virtual file edits.
"""

from __future__ import annotations

import re
from typing import List

from .models import TextSliceRequest, TextSliceResult, TextSliceSegment
from .tool_loop import request_confirmation, run_tool_loop

_SLICE_MARKER = "<slice/>"


def apply_text_slice(request: TextSliceRequest) -> TextSliceResult:
    """
    Apply text slicing using a language model.

    :param request: Text slice request.
    :type request: TextSliceRequest
    :return: Text slice result.
    :rtype: TextSliceResult
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
        apply_str_replace=_apply_slice_replace,
    )
    if not result.done:
        if result.last_error:
            raise ValueError(f"Text slice failed: {result.last_error}")
        warnings.append("Text slice reached max rounds without done=true")
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
            apply_str_replace=_apply_slice_replace,
            confirmation_message=_build_empty_confirmation_message(result.text),
        )
        if not confirmation.done:
            if confirmation.last_error:
                raise ValueError(f"Text slice failed: {confirmation.last_error}")
            warnings.append("Text slice confirmation reached max rounds without done=true")
        _validate_preserved_text(original=request.text, marked_up=confirmation.text)
        slices = _extract_slices(marked_up_text=confirmation.text)
        if confirmation.text == request.text:
            warnings.append("Text slice returned no markers; model confirmed single slice")
        if not slices:
            raise ValueError("Text slice produced no slices")
        return TextSliceResult(
            marked_up_text=confirmation.text,
            slices=slices,
            warnings=warnings,
        )
    _validate_preserved_text(original=request.text, marked_up=result.text)
    slices = _extract_slices(marked_up_text=result.text)
    if not slices:
        raise ValueError("Text slice produced no slices")
    return TextSliceResult(marked_up_text=result.text, slices=slices, warnings=warnings)


def _build_mock_result(request: TextSliceRequest, marked_up_text: str) -> TextSliceResult:
    if marked_up_text == request.text:
        raise ValueError("Text slice produced no markers")
    _validate_preserved_text(original=request.text, marked_up=marked_up_text)
    slices = _extract_slices(marked_up_text=marked_up_text)
    return TextSliceResult(marked_up_text=marked_up_text, slices=slices, warnings=[])


def _apply_slice_replace(text: str, old_str: str, new_str: str) -> str:
    occurrences = text.count(old_str)
    if occurrences == 0:
        raise ValueError("Text slice replacement old_str not found")
    if occurrences > 1:
        raise ValueError("Text slice replacement old_str is not unique")
    _validate_replace_text(old_str, new_str)
    return text.replace(old_str, new_str, 1)


def _validate_replace_text(old_str: str, new_str: str) -> None:
    if _strip_slice_markers(old_str) != _strip_slice_markers(new_str):
        raise ValueError("Text slice replacements may only insert slice markers")


def _validate_preserved_text(*, original: str, marked_up: str) -> None:
    if _strip_slice_markers(marked_up) != original:
        raise ValueError("Text slice edits modified the source text")


def _strip_slice_markers(text: str) -> str:
    return text.replace(_SLICE_MARKER, "")


def _extract_slices(*, marked_up_text: str) -> List[TextSliceSegment]:
    marker_pattern = re.compile(re.escape(_SLICE_MARKER))
    slices: List[TextSliceSegment] = []
    cursor = 0
    original_index = 0

    for match in marker_pattern.finditer(marked_up_text):
        chunk = marked_up_text[cursor : match.start()]
        if chunk:
            slice_end = original_index + len(chunk)
            slices.append(
                TextSliceSegment(
                    index=len(slices) + 1,
                    start_char=original_index,
                    end_char=slice_end,
                    text=chunk,
                )
            )
            original_index = slice_end
        cursor = match.end()

    tail = marked_up_text[cursor:]
    if tail:
        slice_end = original_index + len(tail)
        slices.append(
            TextSliceSegment(
                index=len(slices) + 1,
                start_char=original_index,
                end_char=slice_end,
                text=tail,
            )
        )

    return slices


def _build_empty_confirmation_message(text: str) -> str:
    return (
        "No slice markers were inserted. If the text should remain a single slice, "
        "call done again without changes. Otherwise insert <slice/> markers at the "
        "boundaries of the requested slices.\n"
        "Current text:\n"
        f"---\n{text}\n---"
    )
