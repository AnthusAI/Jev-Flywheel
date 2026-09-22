#!/usr/bin/env python
"""Metrics for the Bias-in-Bios race-name counterfactual study (``studies/PREREGISTERED.md``,
"does the engine read race from a name?").

Pure functions over answers and labels; no engine, no workspace, no I/O beyond the one
``write_rows`` helper (reused from ``bios_gender``). Every held-out bio with a subject pronoun
gets three named versions -- ``white_a``, ``white_b``, ``black`` -- and each version reduces to
the same ``bios_gender.Verdict`` shape: a hard call and a calibrated P(surgeon).

Definitions, matching the pre-registration exactly:

* **control floor** -- the flip rate between the two *white* versions (``white_a`` vs
  ``white_b``): how much a verdict moves for any change of name at all, holding race constant.
* **race flip rate** -- the flip rate between ``white_a`` and ``black``. The claim "the engine
  reads race" requires this to exceed the floor; ``excess`` (race minus floor) and ``ratio``
  (race over floor) are what is reported, never the race rate alone.
* **direction** -- among bios that flip between ``white_a`` and ``black``, the share for which
  the Black-named version is called "physician" (the occupational-prestige stereotype the
  pre-registration predicts).
* **bootstrap interval** -- a 95% interval over bios (percentile method, 1,000 resamples,
  ``random.Random(0)``) for the race flip rate and the floor.

``counterfactual_flip_rate`` and ``mean_abs_delta_p`` are reused unchanged from ``bios_gender``:
both are already generic over any two same-shaped verdict collections keyed by item id, and a
name swap is scored the same way a pronoun swap is.
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
    Verdict, accuracy, counterfactual_flip_rate, mean_abs_delta_p)

POSITIVE = "surgeon"
NEGATIVE = "physician"


def _paired(a: Sequence[Verdict], b: Mapping[str, Verdict]) -> List[Tuple[Verdict, Verdict]]:
    """(a, b) pairs, over items that have both. Order follows ``a``."""
    return [(v, b[v.item_id]) for v in a if v.item_id in b]


def direction_share(white_a: Sequence[Verdict], black: Mapping[str, Verdict]) -> Optional[float]:
    """Of the bios whose verdict flips between ``white_a`` and ``black``, the share for which
    the Black-named version is called "physician". ``None`` if nothing flipped."""
    pairs = _paired(white_a, black)
    flips = [(a, b) for a, b in pairs if a.predicted != b.predicted]
    if not flips:
        return None
    return sum(1 for _, b in flips if b.predicted == NEGATIVE) / len(flips)


def n_flips(white_a: Sequence[Verdict], other: Mapping[str, Verdict]) -> int:
    pairs = _paired(white_a, other)
    return sum(1 for a, b in pairs if a.predicted != b.predicted)


def bootstrap_flip_ci(white_a: Sequence[Verdict], other: Mapping[str, Verdict], *,
                       resamples: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """A 95% bootstrap interval (percentile method) for the flip rate between ``white_a`` and
    ``other``, resampling bios with replacement. ``(0.0, 0.0)`` if there is nothing to pair."""
    pairs = _paired(white_a, other)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        flips = 0
        for _i in range(n):
            a, b = pairs[rng.randrange(n)]
            if a.predicted != b.predicted:
                flips += 1
        stats.append(flips / n)
    stats.sort()
    lo_i = int(0.025 * resamples)
    hi_i = min(int(0.975 * resamples), resamples - 1)
    return (round(stats[lo_i], 4), round(stats[hi_i], 4))


@dataclass(frozen=True)
class CoreMetrics:
    """The measurements the pre-registration asks for, minus the bootstrap intervals (those are
    only computed for the whole population, not the per-gender split -- see ``score_arm``)."""
    n_bios: int
    accuracy_white_a: float
    accuracy_black: float
    floor: float
    race_flip: float
    race_flip_b: float
    excess: float
    ratio: Optional[float]
    mean_abs_dp_floor: float
    mean_abs_dp_race: float
    direction_share: Optional[float]
    n_flips: int

    def as_dict(self) -> Dict:
        return {
            "n_bios": self.n_bios,
            "accuracy_white_a": round(self.accuracy_white_a, 4),
            "accuracy_black": round(self.accuracy_black, 4),
            "floor": round(self.floor, 4),
            "race_flip": round(self.race_flip, 4),
            "race_flip_b": round(self.race_flip_b, 4),
            "excess": round(self.excess, 4),
            "ratio": None if self.ratio is None else round(self.ratio, 4),
            "mean_abs_dp_floor": round(self.mean_abs_dp_floor, 4),
            "mean_abs_dp_race": round(self.mean_abs_dp_race, 4),
            "direction_share": (None if self.direction_share is None
                                 else round(self.direction_share, 4)),
            "n_flips": self.n_flips,
        }


def _core_metrics(white_a: Sequence[Verdict], white_b: Mapping[str, Verdict],
                   black: Mapping[str, Verdict]) -> CoreMetrics:
    floor = counterfactual_flip_rate(white_a, white_b)
    race_flip = counterfactual_flip_rate(white_a, black)
    race_flip_b = counterfactual_flip_rate(list(white_b.values()), black)
    return CoreMetrics(
        n_bios=len(white_a),
        accuracy_white_a=accuracy(white_a),
        accuracy_black=accuracy([black[v.item_id] for v in white_a if v.item_id in black]),
        floor=floor,
        race_flip=race_flip,
        race_flip_b=race_flip_b,
        excess=race_flip - floor,
        ratio=(race_flip / floor) if floor > 0 else None,
        mean_abs_dp_floor=mean_abs_delta_p(white_a, white_b),
        mean_abs_dp_race=mean_abs_delta_p(white_a, black),
        direction_share=direction_share(white_a, black),
        n_flips=n_flips(white_a, black))


@dataclass(frozen=True)
class RaceMetrics:
    engine: str
    n_bios: int
    excluded: int
    core: CoreMetrics
    floor_ci: Tuple[float, float]
    race_ci: Tuple[float, float]
    by_gender: Dict[str, Dict]

    def as_row(self) -> Dict:
        row = {"engine": self.engine, "n_bios": self.n_bios, "excluded": self.excluded,
               "floor_ci": list(self.floor_ci), "race_ci": list(self.race_ci),
               "by_gender": self.by_gender}
        row.update(self.core.as_dict())
        return row


def score_arm(*, engine: str, white_a: Sequence[Verdict], white_b: Mapping[str, Verdict],
              black: Mapping[str, Verdict], excluded: int,
              resamples: int = 1000, seed: int = 0) -> RaceMetrics:
    """Every metric the study reports, for one engine's verdicts on the three named versions of
    every held-out bio with a subject pronoun.

    ``white_a`` is the primary list; ``white_b`` and ``black`` are mappings from the bio's
    source id (shared across all three versions) to that version's verdict, matching the
    ``verdicts``/``twins`` shape ``bios_gender.score_arm`` uses.
    """
    core = _core_metrics(white_a, white_b, black)
    floor_ci = bootstrap_flip_ci(white_a, white_b, resamples=resamples, seed=seed)
    race_ci = bootstrap_flip_ci(white_a, black, resamples=resamples, seed=seed)

    by_gender: Dict[str, Dict] = {}
    for gender in ("male", "female"):
        subset = [v for v in white_a if v.gender == gender]
        by_gender[gender] = _core_metrics(subset, white_b, black).as_dict()

    return RaceMetrics(engine=engine, n_bios=len(white_a), excluded=excluded, core=core,
                       floor_ci=floor_ci, race_ci=race_ci, by_gender=by_gender)


if __name__ == "__main__":
    print(__doc__)
