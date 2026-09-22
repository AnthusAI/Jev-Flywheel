"""Gender-swapped counterfactuals: the same bio with the pronouns flipped.

The test this module supports is causal rather than correlational. A gap in recall between
men's and women's bios can come from the bios being written differently; a change in an
engine's answer when nothing but the pronouns change cannot. So every held-out item is asked
about twice, once as written and once through ``swap_gender``, and a *flip* is an item whose
verdict differs between the two.

The rule is deliberately small and fixed before any run (see ``studies/PREREGISTERED.md``):
personal pronouns, the reflexives, and a short list of gendered role nouns. Names are handled
separately, by ``redact_names`` below (see its docstring for why): the flip rate ``swap_gender``
alone produces is a **lower bound** on an engine's sensitivity to gender, not an estimate of it,
because a first name left in place is itself a gender cue.

The one genuinely ambiguous token is ``her``, which is both the object pronoun (``asked her``)
and the possessive (``her research``). The rule treats it as possessive when the next token is
a plain word and as the object form when it ends the sentence or is followed by punctuation,
a preposition, or a determiner. That is wrong occasionally; each swap records how many
``her`` tokens it resolved so the noise is visible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_NAME_LIST = Path(__file__).resolve().parents[1] / "fixtures" / "bios" / "first_names.txt"

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
    "mr": "ms", "ms": "mr", "mrs": "mr", "miss": "mr",
    "sir": "madam", "madam": "sir",
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

# Phrases whose gendered word is medical content, not the person's gender: a gynaecologist's
# bio is about women whatever the doctor's gender. Tokens inside these spans are left alone.
_PROTECTED = re.compile(r"\b(?:wo)?men'?s (?:health|medicine|hospital|clinic|center|centre)\b", re.I)


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
    protected = set()
    position = 0
    spans = [m.span() for m in _PROTECTED.finditer(text)]
    for i, token in enumerate(tokens):
        if any(start <= position < end for start, end in spans):
            protected.add(i)
        position += len(token)
    out: List[str] = []
    swapped = her_resolved = 0
    words: List[Tuple[int, str]] = [(i, t) for i, t in enumerate(tokens) if t[:1].isalpha()]
    next_word = {i: words[k + 1][1] for k, (i, _) in enumerate(words) if k + 1 < len(words)}
    for i, token in enumerate(tokens):
        lower = token.lower()
        if i in protected:
            out.append(token)
        elif lower == "her":
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


@dataclass(frozen=True)
class Redaction:
    text: str
    redacted: int      # how many tokens were replaced with "[name]"


@lru_cache(maxsize=4)
def _load_names(path: str) -> frozenset:
    """The committed first-name list, lower-cased. Cached per path so a batch job that redacts
    thousands of bios reads the file once."""
    names = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.lower())
    return frozenset(names)


@lru_cache(maxsize=1)
def _load_nlp():
    """The spaCy pipeline, loaded once per process. Imported lazily so the rest of this module,
    and everything that only calls ``swap_gender``, works without spaCy installed."""
    import spacy
    return spacy.load("en_core_web_sm")


def _redact_doc(doc, names: frozenset) -> Redaction:
    """Replace every token that is both inside a ``PERSON`` entity span and on ``names`` with
    ``[name]``, preserving the token's own leading/trailing whitespace and punctuation. Because
    spaCy already splits a possessive off as its own ``'s`` token (``Alysson's`` -> ``Alysson``,
    ``'s``), redacting only the name token and leaving the ``'s`` token untouched is enough to
    turn ``Alysson's`` into ``[name]'s`` without any special-casing here."""
    person_token_ids = {tok.i for ent in doc.ents if ent.label_ == "PERSON" for tok in ent}
    out: List[str] = []
    redacted = 0
    for tok in doc:
        if tok.i in person_token_ids and tok.text.lower() in names:
            out.append("[name]")
            redacted += 1
        else:
            out.append(tok.text)
        out.append(tok.whitespace_)
    return Redaction("".join(out), redacted)


def redact_names(text: str, *, name_list: Path = DEFAULT_NAME_LIST) -> Redaction:
    """Replace first names spaCy tags as ``PERSON`` and that are on ``name_list`` with
    ``[name]``. See ``redact_names_batch`` for many texts at once -- it is the one to use for
    a corpus, since ``nlp.pipe`` batches spaCy's own work across documents.

    A token counts as a redactable name only when *both* signals agree: spaCy's NER says it is
    part of a person's name, and the token is on the committed US-first-name list. Either signal
    alone over-redacts -- spaCy's ``PERSON`` label catches institutions ("Baylor College") and
    plans, and the name list alone catches surnames and hospital names that happen to share a
    first name ("Mercy Hospital"). The intersection is what removes the gender cue without also
    eating the rest of the bio.
    """
    nlp = _load_nlp()
    names = _load_names(str(name_list))
    return _redact_doc(nlp(text), names)


def redact_names_batch(texts: Sequence[str], *, name_list: Path = DEFAULT_NAME_LIST,
                        batch_size: int = 200) -> List[Redaction]:
    """``redact_names`` for many texts, batched through ``nlp.pipe`` for speed (about 8s per
    1,000 bios on the corpus this study uses, vs. calling ``redact_names`` in a loop)."""
    nlp = _load_nlp()
    names = _load_names(str(name_list))
    return [_redact_doc(doc, names) for doc in nlp.pipe(texts, batch_size=batch_size)]
