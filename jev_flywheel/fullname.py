"""Full-name counterfactuals: the same bio with a first *and* last name inserted everywhere
the person is named, not just once.

``studies/PREREGISTERED.md``'s second race attempt ("race from a full name, second attempt")
diagnoses the first attempt's one-token pronoun swap as too weak an instrument and fixes it by
recurring the name cue the way a real bio would: the bio's first subject pronoun becomes the
full name, every ``[name]`` placeholder redaction left behind becomes the first name, and every
surname token spaCy still tags as part of a ``PERSON`` span (redaction only ever removed first
names, so a token like the "Moyer" in "Dr. Moyer" survives it) becomes the drawn surname.

Three independent insertion points, so three independent counters: a bio that has none of them
(no subject pronoun, no placeholder, no surname span) cannot carry a name at all and
``apply_full_name`` returns ``None`` for it, matching ``jev_flywheel.names.insert_name``'s
exclude-and-count rule for the first race study.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from jev_flywheel.names import _SUBJECT_PRONOUN

_PLACEHOLDER = "[name]"


@dataclass(frozen=True)
class FullNameResult:
    """``apply_full_name``'s full accounting: the text, and which insertion kinds fired."""
    text: Optional[str]
    pronoun: bool         # the first subject pronoun was replaced
    placeholders: int     # count of "[name]" placeholders replaced with the first name
    surnames: int         # count of PERSON-span tokens replaced with the surname


def _person_token_ids(doc) -> set:
    return {tok.i for ent in doc.ents if ent.label_ == "PERSON" for tok in ent}


def apply_full_name_full(text: str, first: str, last: str) -> FullNameResult:
    """``apply_full_name``, but returning the full accounting of which insertion kinds fired.

    Runs spaCy once, on the original (unmodified) text, so the PERSON spans it finds are never
    contaminated by a name this function itself inserted. A single left-to-right pass over that
    text's tokens then emits, per token: the full name at the first subject pronoun, the first
    name at each ``[name]`` placeholder (spaCy splits the literal string into three tokens,
    ``[``, ``name``, ``]``, handled here as one unit), the surname at every other PERSON-span
    token, and the token's own text everywhere else.
    """
    from jev_flywheel.counterfactual import _load_nlp  # lazy: matches redact_names' own import

    match = _SUBJECT_PRONOUN.search(text)
    pronoun_start = match.start() if match else None
    pronoun_end = match.end() if match else None

    nlp = _load_nlp()
    doc = nlp(text)
    person_ids = _person_token_ids(doc)

    pronoun_token_i = None
    if pronoun_start is not None:
        span = doc.char_span(pronoun_start, pronoun_end)
        if span is not None and len(span) == 1:
            pronoun_token_i = span[0].i

    out = []
    pronoun_fired = False
    placeholders = 0
    surnames = 0
    n = len(doc)
    i = 0
    while i < n:
        tok = doc[i]
        if tok.i == pronoun_token_i:
            out.append(first + " " + last)
            out.append(tok.whitespace_)
            pronoun_fired = True
            i += 1
            continue
        if (tok.text == "[" and i + 2 < n and doc[i + 1].text == "name"
                and doc[i + 2].text == "]" and tok.whitespace_ == ""
                and doc[i + 1].whitespace_ == ""):
            out.append(first)
            out.append(doc[i + 2].whitespace_)
            placeholders += 1
            i += 3
            continue
        if tok.i in person_ids:
            out.append(last)
            out.append(tok.whitespace_)
            surnames += 1
            i += 1
            continue
        out.append(tok.text)
        out.append(tok.whitespace_)
        i += 1

    if not pronoun_fired and placeholders == 0 and surnames == 0:
        return FullNameResult(None, False, 0, 0)
    return FullNameResult("".join(out), pronoun_fired, placeholders, surnames)


def apply_full_name(text: str, first: str, last: str) -> Optional[str]:
    """Replace the first subject pronoun with ``first + " " + last``, every later ``[name]``
    placeholder with ``first``, and every surname token inside a spaCy ``PERSON`` span (tokens
    that are not ``[name]``) with ``last``. ``None`` if there is no insertion point at all.
    """
    return apply_full_name_full(text, first, last).text
