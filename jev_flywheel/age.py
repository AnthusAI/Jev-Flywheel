"""Age counterfactuals: does the engine read a stated age?

Age travels with experience, and experience is a legitimate input to almost any decision about
a professional. A model that answers differently for "thirty years in practice" than for
"three" is not biased. So the counterfactual must move the person's age while holding their
experience fixed, which rules out shifting the years already in the bio (that moves both) and
rules out relying on natural cues.

The cue is therefore **inserted**, at the bio's first subject pronoun (the same rule
``jev_flywheel.names.insert_name`` uses): "At 61, he is currently researching..." against "At
34, he is currently researching...". When the pronoun opens a sentence, "At {age}, " is
prepended and the pronoun itself is lower-cased ("He" -> "he"); when the pronoun is
mid-sentence, ", at {age}," is inserted before it and the pronoun is left alone. Which case
fired is recorded, because the two read differently and the study reports them.

A bio is eligible for the counterfactual only if it has a subject pronoun, states no year
before 2000, and states no duration of ten or more years -- otherwise the young age (34/35)
would contradict the bio's own text. See ``studies/PREREGISTERED.md``, "does the engine read
age?" for the full rule and the pre-registered predictions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from jev_flywheel.names import _SUBJECT_PRONOUN

# Any 4-digit year from 1950 through 2029; a bio is disqualified only if one it contains is
# below 2000 -- years from 2000 on do not contradict a young inserted age.
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")

# A stated duration: a number (optionally "+"), optionally introduced by "over"/"more than"/
# "nearly"/"almost", followed by "year(s)". The qualifier is not required to match -- "30
# years" disqualifies exactly as "over 30 years" does.
_DURATION = re.compile(
    r"\b(?:(?:over|more\s+than|nearly|almost)\s+)?(\d+)\+?\s+years?\b", re.IGNORECASE)

_SENTENCE_END = re.compile(r"[.!?][\"'’”)\]]?\s*$")


@dataclass(frozen=True)
class Insertion:
    """The result of inserting an age into a bio: the new text and which case fired."""
    text: str
    case: str  # "sentence_initial" or "mid_sentence"


def _is_sentence_initial(text: str, start: int) -> bool:
    """Whether the character at ``start`` opens a sentence: either the very start of the text,
    or the text before it (ignoring trailing whitespace) ends with sentence punctuation."""
    prefix = text[:start]
    if prefix.strip() == "":
        return True
    return bool(_SENTENCE_END.search(prefix))


def insert_age(text: str, age: int) -> Optional[Insertion]:
    """Insert ``age`` at the bio's first subject pronoun; ``None`` if it has none.

    Sentence-initial pronoun ("He is currently researching..."): "At {age}, " is prepended and
    the pronoun is lower-cased -- "At 61, he is currently researching...".

    Mid-sentence pronoun ("In 2003 she joined the clinic."): ", at {age}," is inserted before
    it, and the pronoun keeps its own case -- "In 2003, at 61, she joined the clinic."
    """
    match = _SUBJECT_PRONOUN.search(text)
    if match is None:
        return None
    start, end = match.start(), match.end()
    pronoun = match.group()

    if _is_sentence_initial(text, start):
        new_text = text[:start] + f"At {age}, " + pronoun.lower() + text[end:]
        return Insertion(new_text, "sentence_initial")

    prefix = text[:start].rstrip()
    new_text = prefix + f", at {age}, " + pronoun + text[end:]
    return Insertion(new_text, "mid_sentence")


def eligible(text: str) -> bool:
    """Whether a bio can carry the age counterfactual: a subject pronoun to insert at, no year
    before 2000, and no stated duration of ten or more years (either of which could contradict
    a young inserted age)."""
    if _SUBJECT_PRONOUN.search(text) is None:
        return False
    for match in _YEAR.finditer(text):
        if int(match.group(1)) < 2000:
            return False
    for match in _DURATION.finditer(text):
        if int(match.group(1)) >= 10:
            return False
    return True
