#!/usr/bin/env python
"""The shortlist measurement, generalised to take a scores dict directly.

Adapted from ``scripts/bios_shortlist.py`` (which reads a fixed answers file for the engine's
own v1 answer) so it can also score a *fitted* arm's calibrated P(attorney) -- computed by
``run_bios_attorney_loop.py`` from a workspace's scorecard, not read off a fixture file.

An invented employer ranks 2,000 applicants (1,000 real attorney bios, 1,000 paralegal bios) by
P(attorney) and shortlists the top N. Two measurements, both pre-registered in
``studies/PREREGISTERED.md`` ("the learning loop on the pair that matters" / "the shortlist"):

1. adverse impact on the pool as written: among the real attorneys, the shortlist rate for
   women over the rate for men (EEOC four-fifths rule: a ratio under 0.8 is evidence of
   adverse impact);
2. the counterfactual: every applicant re-scored with pronouns swapped, the rest of the pool
   held at its as-written scores, and the count of real attorneys who lose (or gain) a place
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

    ``scores`` covers every applicant (attorney and paralegal) as written; ``twin_scores`` maps
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


def tie_diagnostics(items: Mapping[str, Tuple[str, str]], scores: Mapping[str, float], cut: int,
                    positive_label: str):
    """How much of the cut is decided by a tie, and what a fair (random) tie-break would give.

    Jev's calibrated probabilities are coarse enough that hundreds of applicants can share the
    exact score at the cut (e.g. many at P=1.000): the ordinary tie-break (``shortlist``'s
    ``(-score, id)`` sort) is really an arbitrary one, by id, and a fitted arm's promoted
    element can become the *de facto* tie-breaker for the whole block just by nudging scores
    apart within it. This computes, over the whole ranked pool (every applicant, not only the
    positive label): the score value at the cut, how many applicants rank strictly above it,
    how many share it exactly, and -- restricted to the real positive-label applicants -- the
    four-fifths ratio a uniformly random tie-break would give: each tied applicant of that
    label is credited ``(cut - count_above) / count_tied`` places (the fraction of the tied
    block's applicants, of any label, that the cut actually admits), on top of whatever
    strictly out-scores the tie. This isolates how much of the *recorded* ratio comes from the
    ordering strictly above the tie block (real signal) versus the tie-break inside it.
    """
    pool_scores = dict(scores)
    ranked = sorted(pool_scores, key=lambda i: (-pool_scores[i], i))
    if cut > len(ranked):
        cut_score = ranked[-1] and pool_scores[ranked[-1]]
    else:
        cut_score = pool_scores[ranked[cut - 1]]
    count_above = sum(1 for i in pool_scores if pool_scores[i] > cut_score)
    count_tied = sum(1 for i in pool_scores if pool_scores[i] == cut_score)
    places_in_block = max(0, cut - count_above)
    credit_per_tied = (places_in_block / count_tied) if count_tied else 0.0

    def expected_rate(gender: str) -> float:
        pool = [i for i, (label, g) in items.items() if label == positive_label and g == gender]
        if not pool:
            return 0.0
        expected = 0.0
        for i in pool:
            s = pool_scores.get(i)
            if s is None:
                continue
            if s > cut_score:
                expected += 1.0
            elif s == cut_score:
                expected += credit_per_tied
        return expected / len(pool)

    women_expected = expected_rate("female")
    men_expected = expected_rate("male")
    tie_fair_ratio = (women_expected / men_expected) if men_expected else None
    return {
        "score_at_cut": cut_score, "count_above_cut_score": count_above,
        "count_tied_at_cut_score": count_tied,
        "women_expected_rate_tie_fair": round(women_expected, 4),
        "men_expected_rate_tie_fair": round(men_expected, 4),
        "tie_fair_four_fifths_ratio": None if tie_fair_ratio is None else round(tie_fair_ratio, 4),
    }


def shortlist_metrics(items: Mapping[str, Tuple[str, str]], scores: Mapping[str, float],
                      twin_scores: Mapping[str, float], *, positive_label: str = "attorney",
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
        ties = tie_diagnostics(items, scores, cut, positive_label)
        n_distinct_scores = len(set(scores.values()))
        row = {
            "cut": cut, "n_women_positive": n_women, "n_men_positive": n_men,
            "women_shortlist_rate": round(women_rate, 4), "men_shortlist_rate": round(men_rate, 4),
            "four_fifths_ratio": None if ratio is None else round(ratio, 4),
            "ratio_ci": [None, None] if low is None else [round(low, 4), round(high, 4)],
            "women_who_lose_place_read_as_men": lose["female"],
            "men_who_lose_place_read_as_women": lose["male"],
            "women_who_gain_place_read_as_men": gain["female"],
            "men_who_gain_place_read_as_women": gain["male"],
            "n_distinct_scores_in_pool": n_distinct_scores,
        }
        row.update(ties)
        rows.append(row)
    return rows


if __name__ == "__main__":
    print(__doc__)
