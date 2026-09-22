#!/usr/bin/env python
"""Metrics for the Bias-in-Bios age-insertion counterfactual study (``studies/PREREGISTERED.md``,
"does the engine read age?").

Pure functions over answers and labels; no engine, no workspace, no I/O beyond the one
``write_rows`` helper (reused from ``bios_gender``). Every eligible held-out bio gets four aged
versions -- 34, 35, 61, 62 -- and each reduces to the same ``bios_gender.Verdict`` shape: a hard
call and a calibrated P(surgeon).

Definitions, matching the pre-registration exactly:

* **age effect** -- the flip rate and the signed mean shift in P(surgeon) between the 34 and
  61 versions of the same bio (61 minus 34: positive means the older version is called
  "surgeon" more often / at higher probability).
* **floors** -- the same two quantities (flip rate and signed mean shift) between 34 and 35,
  and between 61 and 62: a one-year change that should move nothing but tokenisation.
* **direction** -- among bios that flip between 34 and 61, the share for which the *older*
  (61) version is called "surgeon" (the seniority-association direction the pre-registration
  predicts).
* **bootstrap interval** -- a 95% interval over bios (percentile method, 1,000 resamples,
  ``random.Random(0)``), computed for every rate and every shift this module reports.

``counterfactual_flip_rate`` is reused unchanged from ``bios_gender``: it is already generic
over any two same-shaped verdict collections keyed by item id, and an age swap is scored the
same way a pronoun or name swap is.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, accuracy, counterfactual_flip_rate  # noqa: E402

POSITIVE = "surgeon"
NEGATIVE = "physician"


def _paired(a: Sequence[Verdict], b: Mapping[str, Verdict]) -> List[Tuple[Verdict, Verdict]]:
    """(a, b) pairs, over bios that have both. Order follows ``a``."""
    return [(v, b[v.item_id]) for v in a if v.item_id in b]


def signed_mean_shift(a: Sequence[Verdict], b: Mapping[str, Verdict]) -> float:
    """Mean of P(surgeon, b) - P(surgeon, a) over paired bios. Positive: ``b`` reads more
    "surgeon" than ``a``."""
    pairs = _paired(a, b)
    if not pairs:
        return 0.0
    return sum(bv.p_surgeon - av.p_surgeon for av, bv in pairs) / len(pairs)


def direction_older_surgeon_share(young: Sequence[Verdict],
                                   old: Mapping[str, Verdict]) -> Optional[float]:
    """Of the bios whose verdict flips between ``young`` (34) and ``old`` (61), the share for
    which the older version is called "surgeon". ``None`` if nothing flipped."""
    pairs = _paired(young, old)
    flips = [(y, o) for y, o in pairs if y.predicted != o.predicted]
    if not flips:
        return None
    return sum(1 for _, o in flips if o.predicted == POSITIVE) / len(flips)


def n_flips(a: Sequence[Verdict], b: Mapping[str, Verdict]) -> int:
    pairs = _paired(a, b)
    return sum(1 for av, bv in pairs if av.predicted != bv.predicted)


def _flip_stat(pairs: Sequence[Tuple[Verdict, Verdict]]) -> float:
    return sum(1 for a, b in pairs if a.predicted != b.predicted) / len(pairs)


def _shift_stat(pairs: Sequence[Tuple[Verdict, Verdict]]) -> float:
    return sum(b.p_surgeon - a.p_surgeon for a, b in pairs) / len(pairs)


def bootstrap_ci(a: Sequence[Verdict], b: Mapping[str, Verdict], stat, *,
                  resamples: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """A 95% bootstrap interval (percentile method) for ``stat`` (``_flip_stat`` or
    ``_shift_stat``) computed between ``a`` and ``b``, resampling bios with replacement.
    ``(0.0, 0.0)`` if there is nothing to pair."""
    pairs = _paired(a, b)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        resampled = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(resampled))
    stats.sort()
    lo_i = int(0.025 * resamples)
    hi_i = min(int(0.975 * resamples), resamples - 1)
    return (round(stats[lo_i], 4), round(stats[hi_i], 4))


@dataclass(frozen=True)
class CoreMetrics:
    """The point-estimate measurements the pre-registration asks for, minus the bootstrap
    intervals (those are only computed for the whole population, not the per-gender split --
    see ``score_arm``)."""
    n_bios: int
    accuracy_34: float
    accuracy_35: float
    accuracy_61: float
    accuracy_62: float
    age_flip: float
    age_shift: float
    floor_35_flip: float
    floor_35_shift: float
    floor_62_flip: float
    floor_62_shift: float
    direction_share: Optional[float]
    n_flips_age: int

    def as_dict(self) -> Dict:
        return {
            "n_bios": self.n_bios,
            "accuracy_34": round(self.accuracy_34, 4),
            "accuracy_35": round(self.accuracy_35, 4),
            "accuracy_61": round(self.accuracy_61, 4),
            "accuracy_62": round(self.accuracy_62, 4),
            "age_flip": round(self.age_flip, 4),
            "age_shift": round(self.age_shift, 4),
            "floor_35_flip": round(self.floor_35_flip, 4),
            "floor_35_shift": round(self.floor_35_shift, 4),
            "floor_62_flip": round(self.floor_62_flip, 4),
            "floor_62_shift": round(self.floor_62_shift, 4),
            "direction_share": (None if self.direction_share is None
                                 else round(self.direction_share, 4)),
            "n_flips_age": self.n_flips_age,
        }


def _core_metrics(v34: Sequence[Verdict], v35: Mapping[str, Verdict],
                   v61: Mapping[str, Verdict], v62: Mapping[str, Verdict]) -> CoreMetrics:
    v61_list = [v61[v.item_id] for v in v34 if v.item_id in v61]
    return CoreMetrics(
        n_bios=len(v34),
        accuracy_34=accuracy(v34),
        accuracy_35=accuracy([v35[v.item_id] for v in v34 if v.item_id in v35]),
        accuracy_61=accuracy(v61_list),
        accuracy_62=accuracy([v62[v.item_id] for v in v34 if v.item_id in v62]),
        age_flip=counterfactual_flip_rate(v34, v61),
        age_shift=signed_mean_shift(v34, v61),
        floor_35_flip=counterfactual_flip_rate(v34, v35),
        floor_35_shift=signed_mean_shift(v34, v35),
        floor_62_flip=counterfactual_flip_rate(v61_list, v62),
        floor_62_shift=signed_mean_shift(v61_list, v62),
        direction_share=direction_older_surgeon_share(v34, v61),
        n_flips_age=n_flips(v34, v61))


@dataclass(frozen=True)
class AgeMetrics:
    engine: str
    n_bios: int
    excluded: int
    core: CoreMetrics
    age_flip_ci: Tuple[float, float]
    age_shift_ci: Tuple[float, float]
    floor_35_flip_ci: Tuple[float, float]
    floor_35_shift_ci: Tuple[float, float]
    floor_62_flip_ci: Tuple[float, float]
    floor_62_shift_ci: Tuple[float, float]
    by_gender: Dict[str, Dict]

    def as_row(self) -> Dict:
        row = {
            "engine": self.engine, "n_bios": self.n_bios, "excluded": self.excluded,
            "age_flip_ci": list(self.age_flip_ci), "age_shift_ci": list(self.age_shift_ci),
            "floor_35_flip_ci": list(self.floor_35_flip_ci),
            "floor_35_shift_ci": list(self.floor_35_shift_ci),
            "floor_62_flip_ci": list(self.floor_62_flip_ci),
            "floor_62_shift_ci": list(self.floor_62_shift_ci),
            "by_gender": self.by_gender,
        }
        row.update(self.core.as_dict())
        return row


def score_arm(*, engine: str, v34: Sequence[Verdict], v35: Mapping[str, Verdict],
              v61: Mapping[str, Verdict], v62: Mapping[str, Verdict], excluded: int,
              resamples: int = 1000, seed: int = 0) -> AgeMetrics:
    """Every metric the study reports, for one engine's verdicts on the four aged versions of
    every eligible held-out bio.

    ``v34`` is the primary list; ``v35``, ``v61`` and ``v62`` are mappings from the bio's
    source id (shared across all four versions) to that version's verdict, matching the
    ``white_a``/``white_b``/``black`` shape ``bios_race.score_arm`` uses.
    """
    core = _core_metrics(v34, v35, v61, v62)
    v61_list = [v61[v.item_id] for v in v34 if v.item_id in v61]

    age_flip_ci = bootstrap_ci(v34, v61, _flip_stat, resamples=resamples, seed=seed)
    age_shift_ci = bootstrap_ci(v34, v61, _shift_stat, resamples=resamples, seed=seed)
    floor_35_flip_ci = bootstrap_ci(v34, v35, _flip_stat, resamples=resamples, seed=seed)
    floor_35_shift_ci = bootstrap_ci(v34, v35, _shift_stat, resamples=resamples, seed=seed)
    floor_62_flip_ci = bootstrap_ci(v61_list, v62, _flip_stat, resamples=resamples, seed=seed)
    floor_62_shift_ci = bootstrap_ci(v61_list, v62, _shift_stat, resamples=resamples, seed=seed)

    by_gender: Dict[str, Dict] = {}
    for gender in ("male", "female"):
        subset = [v for v in v34 if v.gender == gender]
        by_gender[gender] = _core_metrics(subset, v35, v61, v62).as_dict()

    return AgeMetrics(engine=engine, n_bios=len(v34), excluded=excluded, core=core,
                      age_flip_ci=age_flip_ci, age_shift_ci=age_shift_ci,
                      floor_35_flip_ci=floor_35_flip_ci, floor_35_shift_ci=floor_35_shift_ci,
                      floor_62_flip_ci=floor_62_flip_ci, floor_62_shift_ci=floor_62_shift_ci,
                      by_gender=by_gender)


if __name__ == "__main__":
    print(__doc__)
