"""Gender-swapped counterfactuals: the same bio with the pronouns flipped.

The test this module supports is causal rather than correlational. A gap in recall between
men's and women's bios can come from the bios being written differently; a change in an
engine's answer when nothing but the pronouns change cannot. So every held-out item is asked
about twice, once as written and once through ``swap_gender``, and a *flip* is an item whose
verdict differs between the two.

The rule is deliberately small and fixed before any run (see ``studies/PREREGISTERED.md``):
personal pronouns, the reflexives, and a short list of gendered role nouns. Names are left
alone -- the corpus's ``hard_text`` keeps first names in the body of a bio -- so the flip rate
this produces is a **lower bound** on an engine's sensitivity to gender, not an estimate of it.

The one genuinely ambiguous token is ``her``, which is both the object pronoun (``asked her``)
and the possessive (``her research``). The rule treats it as possessive when the next token is
a plain word and as the object form when it ends the sentence or is followed by punctuation,
a preposition, or a determiner. That is wrong occasionally; each swap records how many
``her`` tokens it resolved so the noise is visible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Unambiguous, case-insensitive, whole-word swaps. Both directions are listed so one table
# serves either starting gender.
_PAIRS: Dict[str, str] = {
    "he": "she", "she": "he",
    "him": "her", "his": "her",
    "himself": "herself", "herself": "himself",
    "hers": "his",
    "man": "woman", "woman": "man",
    "men": "women", "women": "men",
    "father": "mother", "mother": "father",
    "husband": "wife", "wife": "husband",
    "son": "daughter", "daughter": "son",
    "brother": "sister", "sister": "brother",
    "mr": "ms", "ms": "mr", "mrs": "mr",
    "boy": "girl", "girl": "boy",
    "male": "female", "female": "male",
    "gentleman": "lady", "lady": "gentleman",
    "king": "queen", "queen": "king",
    "actor": "actress", "actress": "actor",
    "chairman": "chairwoman", "chairwoman": "chairman",
    "spokesman": "spokeswoman", "spokeswoman": "spokesman",
}

# Words after ``her`` that mark it as the object pronoun rather than a possessive.
_OBJECT_FOLLOWERS = {
    "a", "an", "the", "this", "that", "these", "those", "to", "for", "with", "at", "in",
    "on", "of", "by", "from", "as", "and", "or", "but", "into", "about", "after", "before",
    "while", "when", "where", "who", "which", "if", "so", "up", "out", "over", "off",
}

_TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


@dataclass(frozen=True)
class Swap:
    text: str
    swapped: int          # how many tokens changed
    her_resolved: int     # how many ambiguous ``her`` tokens the heuristic decided


def _match_case(source: str, replacement: str) -> str:
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def swap_gender(text: str) -> Swap:
    """Return ``text`` with its gendered pronouns and role nouns flipped.

    The mapping is an involution except for ``her``, whose two readings collapse onto ``him``
    and ``his``; swapping twice is therefore not guaranteed to return the original text.
    """
    tokens: List[str] = _TOKEN.findall(text)
    out: List[str] = []
    swapped = her_resolved = 0
    words: List[Tuple[int, str]] = [(i, t) for i, t in enumerate(tokens) if t[:1].isalpha()]
    next_word = {i: words[k + 1][1] for k, (i, _) in enumerate(words) if k + 1 < len(words)}
    for i, token in enumerate(tokens):
        lower = token.lower()
        if lower == "her":
            her_resolved += 1
            following = next_word.get(i, "").lower()
            ends_clause = i + 1 >= len(tokens) or not tokens[i + 1].isspace() or following == ""
            possessive = not ends_clause and following not in _OBJECT_FOLLOWERS
            out.append(_match_case(token, "his" if possessive else "him"))
            swapped += 1
        elif lower in _PAIRS:
            out.append(_match_case(token, _PAIRS[lower]))
            swapped += 1
        else:
            out.append(token)
    return Swap("".join(out), swapped, her_resolved)
