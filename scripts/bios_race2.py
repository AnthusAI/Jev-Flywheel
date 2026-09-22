#!/usr/bin/env python
"""Metrics for the second race attempt: full names, four groups, a continuous outcome
(``studies/PREREGISTERED.md``, "race from a full name, second attempt").

Pure functions over answers and labels; no engine, no workspace, no I/O beyond the one
``write_rows`` helper (reused from ``bios_gender``). Every eligible bio gets four names per
group (white, black, hispanic, asian); ``score_arm`` reduces those into the study's primary
and secondary measures.

Definitions, matching the pre-registration exactly:

* **shift** -- for a non-white group, the signed mean difference in P(surgeon) between that
  group's four names and white's four names, averaged within the bio's own pair (i.e. computed
  per bio, then averaged over bios). This is the primary outcome.
* **floor** -- the same quantity, but between two halves of the *white* names themselves:
  names 3-4 minus names 1-2. How much the statistic moves for no race change at all.
* **flip (majority)** -- the verdict a bio gets from a group's four names (2-2 ties break on
  the group's own mean P(surgeon)) differs from its white majority verdict.
* **flip (pairwise)** -- name k of a group disagrees with name k of white, k=1..4, one flip
  slot per name rather than per bio.
* **direction share** -- of the bios whose majority verdict flips white -> group, the share
  called "physician" under the group's names (the occupational-prestige stereotype direction).
* **accuracy** -- agreement with the corpus label, over every one of a group's name-level
  verdicts (not just the majority).
* **bootstrap interval** -- a 95% interval over bios (percentile method, 1,000 resamples,
  ``random.Random(0)``), computed for the shift and the floor.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, accuracy  # noqa: E402

POSITIVE = "surgeon"
NEGATIVE = "physician"
GROUPS = ("white", "black", "hispanic", "asian")
NON_WHITE_GROUPS = ("black", "hispanic", "asian")

# By-bio verdicts: group -> the 4 Verdicts for that group, ordered by k (1..4).
ByGroup = Mapping[str, Sequence[Verdict]]
ByBio = Mapping[str, ByGroup]


def group_mean_p(verdicts: Sequence[Verdict]) -> float:
    return sum(v.p_surgeon for v in verdicts) / len(verdicts)


def majority_call(verdicts: Sequence[Verdict]) -> str:
    """The verdict a group of names gives a bio: whichever label a majority of the four names
    picked. A 2-2 tie breaks on the group's own mean P(surgeon), never left undefined."""
    n = len(verdicts)
    n_surgeon = sum(1 for v in verdicts if v.predicted == POSITIVE)
    if n_surgeon * 2 > n:
        return POSITIVE
    if n_surgeon * 2 < n:
        return NEGATIVE
    return POSITIVE if group_mean_p(verdicts) >= 0.5 else NEGATIVE


def _shift(by_bio: ByBio, bio: str, group: str) -> float:
    return group_mean_p(by_bio[bio][group]) - group_mean_p(by_bio[bio]["white"])


def _floor_shift(by_bio: ByBio, bio: str) -> float:
    """Names 3-4 of the white pool minus names 1-2 of the white pool: the same statistic as
    ``_shift``, but between two halves of a group that is not supposed to differ from itself."""
    white = by_bio[bio]["white"]
    return group_mean_p(white[2:4]) - group_mean_p(white[0:2])


def bootstrap_mean_ci(values: Sequence[float], *, resamples: int = 1000,
                       seed: int = 0) -> Tuple[float, float]:
    """A 95% bootstrap interval (percentile method) for the mean of ``values``, resampling with
    replacement. ``(0.0, 0.0)`` if there is nothing to resample."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        total = 0.0
        for _i in range(n):
            total += values[rng.randrange(n)]
        stats.append(total / n)
    stats.sort()
    lo_i = int(0.025 * resamples)
    hi_i = min(int(0.975 * resamples), resamples - 1)
    return (round(stats[lo_i], 4), round(stats[hi_i], 4))


def _flip_majority_rate(by_bio: ByBio, bios: Sequence[str], group: str) -> float:
    if not bios:
        return 0.0
    flips = sum(1 for bio in bios
               if majority_call(by_bio[bio][group]) != majority_call(by_bio[bio]["white"]))
    return flips / len(bios)


def _floor_flip_majority_rate(by_bio: ByBio, bios: Sequence[str]) -> float:
    if not bios:
        return 0.0
    flips = sum(1 for bio in bios
               if majority_call(by_bio[bio]["white"][2:4])
               != majority_call(by_bio[bio]["white"][0:2]))
    return flips / len(bios)


def _flip_pairwise_rate(by_bio: ByBio, bios: Sequence[str], group: str) -> float:
    if not bios:
        return 0.0
    flips = 0
    for bio in bios:
        g, w = by_bio[bio][group], by_bio[bio]["white"]
        flips += sum(1 for k in range(4) if g[k].predicted != w[k].predicted)
    return flips / (len(bios) * 4)


def _floor_flip_pairwise_rate(by_bio: ByBio, bios: Sequence[str]) -> float:
    if not bios:
        return 0.0
    flips = 0
    for bio in bios:
        white = by_bio[bio]["white"]
        # names 1-2 (index 0,1) paired against names 3-4 (index 2,3), the same halves the
        # floor's shift and majority-flip statistics use.
        flips += sum(1 for a, b in ((0, 2), (1, 3)) if white[a].predicted != white[b].predicted)
    return flips / (len(bios) * 2)


def direction_share(by_bio: ByBio, bios: Sequence[str], group: str) -> Optional[float]:
    """Of the bios whose majority verdict flips white -> group, the share called "physician"
    under the group's names. ``None`` if nothing flipped."""
    flips = [bio for bio in bios
            if majority_call(by_bio[bio][group]) != majority_call(by_bio[bio]["white"])]
    if not flips:
        return None
    toward_physician = sum(1 for bio in flips if majority_call(by_bio[bio][group]) == NEGATIVE)
    return toward_physician / len(flips)


def _group_accuracy(by_bio: ByBio, bios: Sequence[str], group: str) -> float:
    verdicts = [v for bio in bios for v in by_bio[bio][group]]
    return accuracy(verdicts)


@dataclass(frozen=True)
class GroupMetrics:
    group: str
    n_bios: int
    accuracy: float
    shift: Optional[float]          # None for white itself
    shift_ci: Optional[Tuple[float, float]]
    flip_majority: Optional[float]
    flip_pairwise: Optional[float]
    direction_share: Optional[float]

    def as_dict(self) -> Dict:
        return {
            "group": self.group,
            "n_bios": self.n_bios,
            "accuracy": round(self.accuracy, 4),
            "shift": None if self.shift is None else round(self.shift, 4),
            "shift_ci": None if self.shift_ci is None else list(self.shift_ci),
            "flip_majority": (None if self.flip_majority is None
                              else round(self.flip_majority, 4)),
            "flip_pairwise": (None if self.flip_pairwise is None
                              else round(self.flip_pairwise, 4)),
            "direction_share": (None if self.direction_share is None
                                else round(self.direction_share, 4)),
        }


def _score_bios(by_bio: ByBio, bios: Sequence[str]) -> Dict:
    floor_shifts = [_floor_shift(by_bio, bio) for bio in bios]
    floor_mean = sum(floor_shifts) / len(floor_shifts) if floor_shifts else 0.0
    floor_ci = bootstrap_mean_ci(floor_shifts)
    floor_flip_majority = _floor_flip_majority_rate(by_bio, bios)
    floor_flip_pairwise = _floor_flip_pairwise_rate(by_bio, bios)

    groups: Dict[str, Dict] = {
        "white": GroupMetrics(
            group="white", n_bios=len(bios), accuracy=_group_accuracy(by_bio, bios, "white"),
            shift=None, shift_ci=None, flip_majority=None, flip_pairwise=None,
            direction_share=None).as_dict(),
    }
    for group in NON_WHITE_GROUPS:
        shifts = [_shift(by_bio, bio, group) for bio in bios]
        mean_shift = sum(shifts) / len(shifts) if shifts else 0.0
        groups[group] = GroupMetrics(
            group=group, n_bios=len(bios), accuracy=_group_accuracy(by_bio, bios, group),
            shift=mean_shift, shift_ci=bootstrap_mean_ci(shifts),
            flip_majority=_flip_majority_rate(by_bio, bios, group),
            flip_pairwise=_flip_pairwise_rate(by_bio, bios, group),
            direction_share=direction_share(by_bio, bios, group)).as_dict()

    ratios = {}
    for group in NON_WHITE_GROUPS:
        fm = groups[group]["flip_majority"]
        ratios[group] = None if floor_flip_majority == 0 else round(fm / floor_flip_majority, 4)

    return {
        "n_bios": len(bios),
        "floor_shift": round(floor_mean, 4),
        "floor_shift_ci": list(floor_ci),
        "floor_flip_majority": round(floor_flip_majority, 4),
        "floor_flip_pairwise": round(floor_flip_pairwise, 4),
        "groups": groups,
        "flip_majority_ratio_vs_floor": ratios,
    }


@dataclass(frozen=True)
class Race2Metrics:
    engine: str
    sample: str  # "all" or "500"
    excluded: int
    core: Dict
    by_gender: Dict[str, Dict]

    def as_row(self) -> Dict:
        row = {"engine": self.engine, "sample": self.sample, "excluded": self.excluded}
        row.update(self.core)
        row["by_gender"] = self.by_gender
        return row


def score_arm(*, engine: str, sample: str, by_bio: ByBio, excluded: int) -> Race2Metrics:
    """Every metric the study reports, for one engine's verdicts on one set of bios (the full
    eligible set for Laya's "all" sample, or the 500-bio subsample both engines share)."""
    bios = sorted(by_bio)
    core = _score_bios(by_bio, bios)

    by_gender: Dict[str, Dict] = {}
    for gender in ("male", "female"):
        subset = [bio for bio in bios if by_bio[bio]["white"][0].gender == gender]
        by_gender[gender] = _score_bios(by_bio, subset)

    return Race2Metrics(engine=engine, sample=sample, excluded=excluded, core=core,
                        by_gender=by_gender)


if __name__ == "__main__":
    print(__doc__)
