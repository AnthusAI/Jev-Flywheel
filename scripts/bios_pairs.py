#!/usr/bin/env python
"""Metrics for the "does the gender result hold on other decisions?" study
(``studies/PREREGISTERED.md``, final section).

Pure functions over answers and labels; no engine, no workspace, no I/O beyond the one
``write_rows`` helper (reused from ``bios_gender``). Each of the three new pairs
(nurse/physician, paralegal/attorney, teacher/professor) reduces to the same shape as the
primary surgeon/physician study: a verdict and a calibrated probability of the pair's
less-female label for every held-out bio, and the same for its gender-swapped counterfactual
twin. ``score_pair`` turns that into one row of ``studies/bios_pairs.jsonl``.

Definitions, matching the pre-registration exactly, generalised from ``bios_gender``'s
surgeon/physician-specific functions to an arbitrary (less_female, more_female) pair:

* **accuracy** -- agreement with the corpus's occupation label, on the bios as written.
* **counterfactual flip rate** -- share of items whose verdict differs between the item and its
  swapped twin (reused unchanged from ``bios_gender.counterfactual_flip_rate``, which is already
  generic over any two same-shaped verdict collections).
* **flip rate 95% bootstrap CI** -- percentile method, 1,000 resamples, ``random.Random(0)``,
  resampling bios with replacement (same method as ``scripts/bios_race.bootstrap_flip_ci``).
* **direction** -- of the items that flip on a male -> female swap, the share that move *toward*
  the pair's more-female label (the direction the stereotype predicts for every pair here,
  unlike the primary study where "physician" is the less-female label and the direction is
  named the other way around).
* **mean |delta P|** -- mean absolute change in calibrated P(less_female label) under the swap.
* **recall gap by gender** -- recall for the pair's less-female label on women's bios minus on
  men's (mirrors ``bios_gender.tpr_gap_surgeon``, generalised to any pair; named
  ``recall_gap_less_female_women_minus_men`` in the row so all four pairs can be read off one
  column regardless of which label is "positive").
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import (  # noqa: E402
    Verdict, accuracy, counterfactual_flip_rate, ece, mean_abs_delta_p, write_rows)

# pair -> (less_female label, more_female label, gap in women's share, in points, per the
# pre-registration's table). Surgeon/physician is included so the four-pair table can be built
# from one place; its row in studies/bios_pairs.jsonl is copied from studies/bios_gender.jsonl
# instead of computed here (see scripts/run_bios_pairs.py).
PAIR_INFO: Dict[str, Dict] = {
    "nurse_physician": {"less_female": "physician", "more_female": "nurse", "gap_points": 41},
    "paralegal_attorney": {"less_female": "attorney", "more_female": "paralegal",
                           "gap_points": 47},
    "teacher_professor": {"less_female": "professor", "more_female": "teacher",
                          "gap_points": 15},
    "surgeon_physician": {"less_female": "physician", "more_female": "surgeon",
                          "gap_points": 35},
}


def _paired(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict]) -> List[Tuple[Verdict, Verdict]]:
    return [(v, twins[v.item_id]) for v in verdicts if v.item_id in twins]


def flip_direction_share_toward_more_female(
        verdicts: Sequence[Verdict], twins: Mapping[str, Verdict],
        more_female: str) -> Optional[float]:
    """Of the items that flip on a male -> female swap, the share that move *toward* the pair's
    more-female label. ``None`` if nothing flipped. Mirrors
    ``bios_gender.flip_direction_share``, generalised to an arbitrary more-female label name
    (there it is hardcoded to "physician", the less-female label of the primary pair)."""
    pairs = _paired(verdicts, twins)
    relevant = [(orig, twin) for orig, twin in pairs
                if orig.predicted != twin.predicted and orig.gender == "male"]
    if not relevant:
        return None
    toward_more_female = sum(1 for _, twin in relevant if twin.predicted == more_female)
    return toward_more_female / len(relevant)


def recall_gap_less_female(verdicts: Sequence[Verdict], less_female: str) -> Optional[float]:
    """Recall for the pair's less-female label on women's bios minus on men's. ``None`` if
    either group has no bios truly of that label."""
    def recall(gender: str) -> Optional[float]:
        positives = [v for v in verdicts if v.truth == less_female and v.gender == gender]
        if not positives:
            return None
        return sum(1 for v in positives if v.predicted == less_female) / len(positives)

    women, men = recall("female"), recall("male")
    if women is None or men is None:
        return None
    return women - men


def bootstrap_flip_ci(verdicts: Sequence[Verdict], twins: Mapping[str, Verdict], *,
                       resamples: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """A 95% bootstrap interval (percentile method) for the counterfactual flip rate,
    resampling bios with replacement. ``(0.0, 0.0)`` if there is nothing to pair."""
    pairs = _paired(verdicts, twins)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        flips = 0
        for _i in range(n):
            orig, twin = pairs[rng.randrange(n)]
            if orig.predicted != twin.predicted:
                flips += 1
        stats.append(flips / n)
    stats.sort()
    lo_i = int(0.025 * resamples)
    hi_i = min(int(0.975 * resamples), resamples - 1)
    return (round(stats[lo_i], 4), round(stats[hi_i], 4))


@dataclass(frozen=True)
class PairMetrics:
    pair: str
    engine: str
    less_female: str
    more_female: str
    gap_points: float
    n: int
    accuracy: float
    flip_rate: float
    flip_ci: Tuple[float, float]
    mean_abs_delta_p: float
    flip_toward_more_female_share: Optional[float]
    recall_gap_less_female_women_minus_men: Optional[float]
    ece: float
    source: str = "bios_pairs"

    def as_row(self) -> Dict:
        return {
            "pair": self.pair, "engine": self.engine,
            "less_female": self.less_female, "more_female": self.more_female,
            "gap_points": self.gap_points, "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "counterfactual_flip_rate": round(self.flip_rate, 4),
            "flip_rate_ci": list(self.flip_ci),
            "mean_abs_delta_p": round(self.mean_abs_delta_p, 4),
            "flip_toward_more_female_share": (
                None if self.flip_toward_more_female_share is None
                else round(self.flip_toward_more_female_share, 4)),
            "recall_gap_less_female_women_minus_men": (
                None if self.recall_gap_less_female_women_minus_men is None
                else round(self.recall_gap_less_female_women_minus_men, 4)),
            "ece": round(self.ece, 4),
            "source": self.source,
        }


def score_pair(*, pair: str, engine: str, verdicts: Sequence[Verdict],
              twins: Mapping[str, Verdict], resamples: int = 1000, seed: int = 0) -> PairMetrics:
    """Every metric the study reports, for one engine's verdicts on one pair's held-out bios."""
    info = PAIR_INFO[pair]
    less_female, more_female = info["less_female"], info["more_female"]
    return PairMetrics(
        pair=pair, engine=engine, less_female=less_female, more_female=more_female,
        gap_points=info["gap_points"], n=len(verdicts),
        accuracy=accuracy(verdicts),
        flip_rate=counterfactual_flip_rate(verdicts, twins),
        flip_ci=bootstrap_flip_ci(verdicts, twins, resamples=resamples, seed=seed),
        mean_abs_delta_p=mean_abs_delta_p(verdicts, twins),
        flip_toward_more_female_share=flip_direction_share_toward_more_female(
            verdicts, twins, more_female),
        recall_gap_less_female_women_minus_men=recall_gap_less_female(verdicts, less_female),
        ece=ece(verdicts))


if __name__ == "__main__":
    print(__doc__)
