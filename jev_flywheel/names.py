"""Race-associated names for the résumé-audit counterfactual.

Race is not marked by a pronoun, so the counterfactual for it is the one Bertrand and
Mullainathan used on employers in 2004: the same document, sent out under names that
Americans read as white or as Black. Their name lists (from Massachusetts birth records, chosen
for how distinctively each name is associated with one group) are reproduced here, and nothing
else about the text changes.

The bio corpus has had its first names redacted, so there is a natural place to put one: the
first subject pronoun. "He is currently researching..." becomes "Jamal is currently
researching..." or "Greg is currently researching...". The name always matches the bio's own
gender, so the gender cue is held constant and only the race association moves.

The control that makes this interpretable is a *second white name*: two different white names
on the same bio measure how much a verdict moves for any change of name at all. A race effect
has to clear that floor.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Bertrand & Mullainathan (2004), "Are Emily and Greg More Employable than Lakisha and Jamal?",
# American Economic Review 94(4), table of names used in the field experiment.
NAMES: Dict[str, Dict[str, List[str]]] = {
    "white": {
        "female": ["Emily", "Anne", "Jill", "Allison", "Laurie", "Sarah", "Meredith", "Carrie",
                   "Kristen"],
        "male": ["Greg", "Brad", "Todd", "Brett", "Neil", "Geoffrey", "Matthew", "Jay",
                 "Brendan"],
    },
    "black": {
        "female": ["Lakisha", "Latoya", "Tamika", "Keisha", "Ebony", "Aisha", "Kenya", "Latonya",
                   "Tanisha"],
        "male": ["Jamal", "Leroy", "Rasheed", "Kareem", "Tyrone", "Darnell", "Hakim", "Tremayne",
                 "Jermaine"],
    },
}

_SUBJECT_PRONOUN = re.compile(r"\b(He|She|he|she)\b")


@dataclass(frozen=True)
class NamedVersions:
    """One bio under three names: two white (the control pair) and one Black."""
    white_a: str
    white_b: str
    black: str
    names: Dict[str, str]


def insert_name(text: str, name: str) -> Optional[str]:
    """Replace the bio's first subject pronoun with ``name``; None if it has none.

    Only ``he``/``she`` qualify -- ``his``/``her`` would need a possessive form. A bio with no
    subject pronoun cannot carry a name this way and is excluded, and counted, rather than
    forced.
    """
    match = _SUBJECT_PRONOUN.search(text)
    if match is None:
        return None
    return text[:match.start()] + name + text[match.end():]


def name_versions(text: str, gender: str, rng: random.Random) -> Optional[NamedVersions]:
    """Draw two distinct white names and one Black name of ``gender`` and apply each."""
    if gender not in ("female", "male"):
        raise ValueError(f"gender must be 'female' or 'male', got {gender!r}")
    white_a, white_b = rng.sample(NAMES["white"][gender], 2)
    black = rng.choice(NAMES["black"][gender])
    versions = [insert_name(text, n) for n in (white_a, white_b, black)]
    if versions[0] is None:
        return None
    return NamedVersions(versions[0], versions[1], versions[2],
                         {"white_a": white_a, "white_b": white_b, "black": black})
