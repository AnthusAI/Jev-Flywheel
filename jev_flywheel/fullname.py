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

The public entry point most callers want is ``apply_full_name``/``apply_full_name_full``, which
run spaCy fresh every call. A study that draws sixteen names per bio should instead call
``analyze_full_name`` once per bio (the only step that runs spaCy) and ``render_full_name`` once
per name pair -- the PERSON spans and the pronoun position depend only on the bio's original
text, never on which name is being tried.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jev_flywheel.names import _SUBJECT_PRONOUN

# A plan segment is (kind, payload). "TEXT" payloads are literal, already-final text (including
# the token's own trailing whitespace). The other three kinds carry only the trailing whitespace
# to emit after the substituted name; what to substitute is supplied at render time.
_TEXT = "TEXT"
_PRONOUN = "PRONOUN"
_PLACEHOLDER = "PLACEHOLDER"
_SURNAME = "SURNAME"


@dataclass(frozen=True)
class FullNamePlan:
    """The result of analyzing one bio's text: where each insertion kind would go, independent
    of which names eventually fill them in. Re-usable across every name pair drawn for a bio."""
    segments: Tuple[Tuple[str, str], ...]
    has_pronoun: bool
    n_placeholders: int
    n_surnames: int

    @property
    def has_insertion_point(self) -> bool:
        return self.has_pronoun or self.n_placeholders > 0 or self.n_surnames > 0


@dataclass(frozen=True)
class FullNameResult:
    """``apply_full_name``'s full accounting: the text, and which insertion kinds fired."""
    text: Optional[str]
    pronoun: bool         # the first subject pronoun was replaced
    placeholders: int     # count of "[name]" placeholders replaced with the first name
    surnames: int         # count of PERSON-span tokens replaced with the surname


def _person_token_ids(doc) -> set:
    return {tok.i for ent in doc.ents if ent.label_ == "PERSON" for tok in ent}


def analyze_full_name(text: str) -> FullNamePlan:
    """Find, once, every place a full name could go in ``text``: the first subject pronoun,
    every ``[name]`` placeholder, and every PERSON-span token that is not a placeholder. This is
    the only step that runs spaCy; ``render_full_name`` reuses its result for as many name pairs
    as a study needs on the same bio.
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

    segments: List[Tuple[str, str]] = []
    has_pronoun = False
    n_placeholders = 0
    n_surnames = 0
    n = len(doc)
    i = 0
    while i < n:
        tok = doc[i]
        if tok.i == pronoun_token_i:
            segments.append((_PRONOUN, tok.whitespace_))
            has_pronoun = True
            i += 1
            continue
        if (tok.text == "[" and i + 2 < n and doc[i + 1].text == "name"
                and doc[i + 2].text == "]" and tok.whitespace_ == ""
                and doc[i + 1].whitespace_ == ""):
            segments.append((_PLACEHOLDER, doc[i + 2].whitespace_))
            n_placeholders += 1
            i += 3
            continue
        if tok.i in person_ids:
            segments.append((_SURNAME, tok.whitespace_))
            n_surnames += 1
            i += 1
            continue
        segments.append((_TEXT, tok.text + tok.whitespace_))
        i += 1

    return FullNamePlan(tuple(segments), has_pronoun, n_placeholders, n_surnames)


def render_full_name(plan: FullNamePlan, first: str, last: str) -> FullNameResult:
    """Fill in a ``FullNamePlan`` with one (first, last) pair. No spaCy call -- pure string
    building -- so this is cheap enough to call once per drawn name."""
    if not plan.has_insertion_point:
        return FullNameResult(None, False, 0, 0)
    out: List[str] = []
    for kind, payload in plan.segments:
        if kind == _TEXT:
            out.append(payload)
        elif kind == _PRONOUN:
            out.append(first + " " + last + payload)
        elif kind == _PLACEHOLDER:
            out.append(first + payload)
        elif kind == _SURNAME:
            out.append(last + payload)
    return FullNameResult("".join(out), plan.has_pronoun, plan.n_placeholders, plan.n_surnames)


def apply_full_name_full(text: str, first: str, last: str) -> FullNameResult:
    """``analyze_full_name`` and ``render_full_name`` in one call, for a single name pair. Runs
    spaCy fresh every call -- prefer the two-phase API when trying several names on one bio."""
    return render_full_name(analyze_full_name(text), first, last)


def apply_full_name(text: str, first: str, last: str) -> Optional[str]:
    """Replace the first subject pronoun with ``first + " " + last``, every later ``[name]``
    placeholder with ``first``, and every surname token inside a spaCy ``PERSON`` span (tokens
    that are not ``[name]``) with ``last``. ``None`` if there is no insertion point at all.
    """
    return apply_full_name_full(text, first, last).text
