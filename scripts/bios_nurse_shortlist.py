#!/usr/bin/env python
"""The shortlist measurement, generalised to take a scores dict directly.

Copied from ``scripts/bios_attorney_shortlist.py`` (owned by the paralegal/attorney agent, not
imported directly per this study's file-ownership rule) with the docstring's example reworded
for the nurse/physician pair; the code is pair-generic already (``positive_label`` is a
parameter), so nothing else changes.

An invented employer ranks 2,000 applicants (1,000 real physician bios, 1,000 nurse bios) by
P(physician) and shortlists the top N. Two measurements, both pre-registered in
``studies/PREREGISTERED.md`` ("the learning loop on nurse vs physician" / "the shortlist"):

1. adverse impact on the pool as written: among the real physicians, the shortlist rate for
   women over the rate for men (EEOC four-fifths rule: a ratio under 0.8 is evidence of
   adverse impact);
2. the counterfactual: every applicant re-scored with pronouns swapped, the rest of the pool
   held at its as-written scores, and the count of real physicians who lose (or gain) a place
   when read as the other gender.
"""
from __future__ import annotations

import random
from typing import Dict, Mapping, Tuple

CUTS = (250, 500, 1000)


def shortlist(scores: Mapping[str, float], cut: int) -> set:
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return set(ranked[:cut])


def four_fifths(items: Mapping[str, Tuple[str, str]], chosen: set, positive_label: str):
    rate = {}
    for gender in ("female", "male"):
        pool = [i for i, (label, g) in items.items() if label == positive_label and g == gender]
        rate[gender] = sum(1 for i in pool if i in chosen) / len(pool) if pool else 0.0
    ratio = rate["female"] / rate["male"] if rate["male"] else None
    return rate["female"], rate["male"], ratio


def bootstrap_ratio(items: Mapping[str, Tuple[str, str]], scores: Mapping[str, float], cut: int,
                    positive_label: str, resamples: int = 1000, seed: int = 0):
    rng = random.Random(seed)
    ids = list(items)
    ratios = []
    for _ in range(resamples):
        sample = [rng.choice(ids) for _ in ids]
        sub_scores = {f"{i}#{k}": scores[i] for k, i in enumerate(sample)}
        sub_items = {f"{i}#{k}": items[i] for k, i in enumerate(sample)}
        chosen = shortlist(sub_scores, cut)
        _, _, ratio = four_fifths(sub_items, chosen, positive_label)
        if ratio is not None:
            ratios.append(ratio)
    ratios.sort()
    if not ratios:
        return (None, None)
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]


def counterfactual(items: Mapping[str, Tuple[str, str]], scores: Mapping[str, float],
                   twin_scores: Mapping[str, float], cut: int, positive_label: str):
    """Each applicant re-scored with pronouns swapped, alone, the rest of the pool as written.

    ``scores`` covers every applicant (physician and nurse) as written; ``twin_scores`` maps
    an applicant's own id to their swapped-twin score (only needed for the positive-label
    applicants that get re-ranked one at a time).
    """
    pool = dict(scores)
    as_written = shortlist(pool, cut)
    lose = {"female": 0, "male": 0}
    gain = {"female": 0, "male": 0}
    for i, (label, gender) in items.items():
        if label != positive_label:
            continue
        altered = dict(pool)
        altered[i] = twin_scores[i]
        now_in = i in shortlist(altered, cut)
        was_in = i in as_written
        if was_in and not now_in:
            lose[gender] += 1
        if not was_in and now_in:
            gain[gender] += 1
    return lose, gain


def shortlist_metrics(items: Mapping[str, Tuple[str, str]], scores: Mapping[str, float],
                      twin_scores: Mapping[str, float], *, positive_label: str = "physician",
                      cuts=CUTS) -> list:
    """One row per cut, in the same shape as ``bios_shortlist.py``'s ``main`` prints/writes."""
    n_women = sum(1 for label, g in items.values() if label == positive_label and g == "female")
    n_men = sum(1 for label, g in items.values() if label == positive_label and g == "male")
    rows = []
    for cut in cuts:
        chosen = shortlist({i: scores[i] for i in items}, cut)
        women_rate, men_rate, ratio = four_fifths(items, chosen, positive_label)
        low, high = bootstrap_ratio(items, scores, cut, positive_label)
        lose, gain = counterfactual(items, scores, twin_scores, cut, positive_label)
        rows.append({
            "cut": cut, "n_women_positive": n_women, "n_men_positive": n_men,
            "women_shortlist_rate": round(women_rate, 4), "men_shortlist_rate": round(men_rate, 4),
            "four_fifths_ratio": None if ratio is None else round(ratio, 4),
            "ratio_ci": [None, None] if low is None else [round(low, 4), round(high, 4)],
            "women_who_lose_place_read_as_men": lose["female"],
            "men_who_lose_place_read_as_women": lose["male"],
            "women_who_gain_place_read_as_men": gain["female"],
            "men_who_gain_place_read_as_women": gain["male"],
        })
    return rows


if __name__ == "__main__":
    print(__doc__)
