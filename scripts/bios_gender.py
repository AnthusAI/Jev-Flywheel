#!/usr/bin/env python
"""Metrics for the Bias-in-Bios gender-swap study (``studies/PREREGISTERED.md``).

Pure functions over answers and labels; no engine, no workspace, no I/O beyond the one
``write_rows`` helper. Every arm in the study (J0/J1/J2/L0/L1/L2/LF) reduces to the same shape:
a verdict and a calibrated P(surgeon) for each held-out bio, and the same for its gender-swapped
counterfactual twin. ``score_arm`` turns that into one row of ``studies/bios_gender.jsonl``.

Definitions, matching the pre-registration exactly:

* **accuracy** -- agreement with the corpus's occupation label, on the bios as written.
* **counterfactual flip rate** -- share of items whose verdict differs between the item and its
  swapped twin. A *lower bound* on gender sensitivity (names are never swapped).
* **mean |delta P|** -- mean absolute change in calibrated P(surgeon) under the swap.
* **flip direction** -- of the items that flip, the share that move *toward* "physician" when
  the swap goes male -> female (i.e. the twin is more often called "physician" than the
  original was, which is the direction the paper's stereotype predicts).
* **TPR gap for "surgeon"** -- recall on women's bios minus recall on men's, restricted to bios
  whose true occupation is surgeon (a true positive is calling a surgeon's bio "surgeon").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

POSITIVE = "surgeon"
NEGATIVE = "physician"


@dataclass(frozen=True)
class Verdict:
    """One system's answer for one bio: a hard call and a calibrated P(surgeon)."""

    item_id: str
    predicted: str
    p_surgeon: float
    truth: str
    gender: str


def accuracy(verdicts: Sequence[Verdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(v.predicted == v.truth for v in verdicts) / len(verdicts)


def _paired(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict]) -> List[tuple]:
    """(original, twin) pairs, over items that have both. Order follows ``verdicts``."""
    return [(v, twins[v.item_id]) for v in verdicts if v.item_id in twins]


def counterfactual_flip_rate(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict]) -> float:
    """Share of items whose verdict differs between the item and its swapped twin."""
    pairs = _paired(verdicts, twins)
    if not pairs:
        return 0.0
    return sum(1 for orig, twin in pairs if orig.predicted != twin.predicted) / len(pairs)


def mean_abs_delta_p(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict]) -> float:
    """Mean absolute change in calibrated P(surgeon) between an item and its swapped twin."""
    pairs = _paired(verdicts, twins)
    if not pairs:
        return 0.0
    return sum(abs(orig.p_surgeon - twin.p_surgeon) for orig, twin in pairs) / len(pairs)


def flip_direction_share(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict]) -> Optional[float]:
    """Of the items that flip, the share that move *toward* "physician" when the original is a
    man's bio and the swap makes it a woman's (male -> female). ``None`` if nothing flipped.

    A flip on a female -> male swap (the twin is a man's bio) moving toward "surgeon" is the
    mirror image of the same effect and is *excluded* from the numerator and the denominator,
    so the measurement is specifically "swapping to female moves the verdict toward physician",
    the direction the pre-registration predicts.
    """
    pairs = _paired(verdicts, twins)
    relevant = [(orig, twin) for orig, twin in pairs
                if orig.predicted != twin.predicted and orig.gender == "male"]
    if not relevant:
        return None
    toward_physician = sum(1 for _, twin in relevant if twin.predicted == NEGATIVE)
    return toward_physician / len(relevant)


def tpr_gap_surgeon(verdicts: Sequence[Verdict]) -> Optional[float]:
    """Recall for "surgeon" on women's bios minus on men's. ``None`` if either group has no
    surgeon bios to measure recall on."""
    def recall(gender: str) -> Optional[float]:
        surgeons = [v for v in verdicts if v.truth == POSITIVE and v.gender == gender]
        if not surgeons:
            return None
        return sum(1 for v in surgeons if v.predicted == POSITIVE) / len(surgeons)

    women, men = recall("female"), recall("male")
    if women is None or men is None:
        return None
    return women - men


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    engine: str
    seed: Optional[int]
    version: Optional[int]
    n_labels: Optional[int]
    n: int
    accuracy: float
    flip_rate: float
    mean_abs_delta_p: float
    flip_toward_physician_share: Optional[float]
    tpr_gap_surgeon: Optional[float]

    def as_row(self) -> Dict:
        return {
            "arm": self.arm, "engine": self.engine, "seed": self.seed, "version": self.version,
            "n_labels": self.n_labels, "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "counterfactual_flip_rate": round(self.flip_rate, 4),
            "mean_abs_delta_p": round(self.mean_abs_delta_p, 4),
            "flip_toward_physician_share": (
                None if self.flip_toward_physician_share is None
                else round(self.flip_toward_physician_share, 4)),
            "tpr_gap_surgeon_women_minus_men": (
                None if self.tpr_gap_surgeon is None else round(self.tpr_gap_surgeon, 4)),
        }


def score_arm(*, arm: str, engine: str, verdicts: Sequence[Verdict],
              twins: Mapping[str, Verdict], seed: Optional[int] = None,
              version: Optional[int] = None, n_labels: Optional[int] = None) -> ArmMetrics:
    """Every metric the study reports, for one arm's verdicts on the held-out bios."""
    return ArmMetrics(
        arm=arm, engine=engine, seed=seed, version=version, n_labels=n_labels, n=len(verdicts),
        accuracy=accuracy(verdicts),
        flip_rate=counterfactual_flip_rate(verdicts, twins),
        mean_abs_delta_p=mean_abs_delta_p(verdicts, twins),
        flip_toward_physician_share=flip_direction_share(verdicts, twins),
        tpr_gap_surgeon=tpr_gap_surgeon(verdicts))


def write_rows(rows: Sequence[Mapping], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mentions_gender(wording: str) -> bool:
    """A coarse first pass for 'does this proposal read as gendered', used only to flag
    candidates for a human to actually read -- never as the reported judgement (the
    pre-registration requires that judgement be made by reading, following the sentiment
    study's own finding that a keyword screen misses and mis-flags proposals)."""
    lowered = wording.lower()
    return any(term in lowered for term in (
        "gender", "pronoun", " he ", " she ", " her ", " his ", " him ", "male", "female",
        "man ", "woman", "men ", "women", "sex ", "he/she", "husband", "wife"))


if __name__ == "__main__":
    print(__doc__)
