#!/usr/bin/env python
"""The shortlist: what a flip rate does to a ranked screen.

    python scripts/bios_shortlist.py            # both pairs, both engines, cuts 250/500/1000

An invented employer ranks 2,000 applicants (1,000 real attorney bios, 1,000 paralegal bios)
by an engine's P(attorney) and shortlists the top N. Two measurements, both pre-registered in
``studies/PREREGISTERED.md`` ("the shortlist"):

1. adverse impact on the pool as written: among the real attorneys, the shortlist rate for
   women over the rate for men (the EEOC four-fifths rule flags a ratio under 0.8);
2. the counterfactual: every applicant re-scored with pronouns swapped, the rest of the pool
   held at its as-written scores, and the count of real attorneys who lose (or gain) a place
   when read as the other gender.

Reads the answers both engines already gave; asks nothing new. Rows to ``studies/bios_shortlist.jsonl``.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Dict

PAIRS = {"paralegal_attorney": "attorney", "nurse_physician": "physician"}
OUT = Path("studies/bios_shortlist.jsonl")
CUTS = (250, 500, 1000)
ENGINES = {"jev": "answers.jsonl.gz", "laya": "answers-laya.jsonl.gz"}


def load_scores(path: Path, positive: str) -> Dict[str, float]:
    scores = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            answer = next(iter(row["answers"].values()))
            scores[row["id"]] = float(answer["probabilities"][positive])
    return scores


def load_items(pair: Path):
    items = {}
    for line in open(pair / "items.jsonl"):
        row = json.loads(line)
        meta = row["metadata"]
        if meta.get("split") != "test":
            continue
        items[row["id"]] = (meta["reference_label"], meta["gender"])
    return items


def shortlist(scores: Dict[str, float], cut: int) -> set:
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return set(ranked[:cut])


def four_fifths(items, chosen: set, positive: str):
    rate = {}
    for gender in ("female", "male"):
        pool = [i for i, (label, g) in items.items() if label == positive and g == gender]
        rate[gender] = sum(1 for i in pool if i in chosen) / len(pool)
    return rate["female"], rate["male"], (rate["female"] / rate["male"] if rate["male"] else None)


def tie_fair(items, scores, cut, positive):
    """The four-fifths ratio under random tie-breaking: a bio tied at the cut counts as the
    share of remaining places its block gets. Jev reports two-decimal probabilities, so a
    top-500 cut can sit inside a block of several hundred bios all at P = 1.00."""
    values = sorted((scores[i] for i in items), reverse=True)
    at_cut = values[cut - 1]
    above = sum(1 for v in values if v > at_cut)
    tied = sum(1 for v in values if v == at_cut)
    share = (cut - above) / tied
    rate = {}
    for gender in ("female", "male"):
        pool = [i for i, (label, g) in items.items() if label == positive and g == gender]
        rate[gender] = sum(1.0 if scores[i] > at_cut else (share if scores[i] == at_cut else 0.0)
                           for i in pool) / len(pool)
    return rate["female"] / rate["male"], above, tied


def bootstrap_ratio(items, scores, cut, positive, resamples=1000, seed=0):
    rng = random.Random(seed)
    ids = list(items)
    ratios = []
    for _ in range(resamples):
        sample = [rng.choice(ids) for _ in ids]
        sub_scores = {f"{i}#{k}": scores[i] for k, i in enumerate(sample)}
        sub_items = {f"{i}#{k}": items[i] for k, i in enumerate(sample)}
        chosen = shortlist(sub_scores, cut)
        _, _, ratio = four_fifths(sub_items, chosen, positive)
        if ratio is not None:
            ratios.append(ratio)
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]


def counterfactual(items, scores, cut, positive):
    """Each applicant re-scored with pronouns swapped, alone, the rest of the pool as written.

    The applicant's swapped score replaces their own in the pool and the pool is re-ranked
    with the same tie-break, so coarse probabilities (many exact 1.0s from Jev) cannot make a
    tie count as a place gained.
    """
    pool = {i: scores[i] for i in items}
    as_written = shortlist(pool, cut)
    lose = {"female": 0, "male": 0}
    gain = {"female": 0, "male": 0}
    for i, (label, gender) in items.items():
        if label != positive:
            continue
        altered = dict(pool)
        altered[i] = scores[f"{i}-swapped"]
        now_in = i in shortlist(altered, cut)
        was_in = i in as_written
        if was_in and not now_in:
            lose[gender] += 1
        if not was_in and now_in:
            gain[gender] += 1
    return lose, gain


def score(items, scores, positive, engine, pair, variant, rows):
    n_women = sum(1 for label, g in items.values() if label == positive and g == "female")
    n_men = sum(1 for label, g in items.values() if label == positive and g == "male")
    for cut in CUTS:
        chosen = shortlist({i: scores[i] for i in items}, cut)
        women_rate, men_rate, ratio = four_fifths(items, chosen, positive)
        low, high = bootstrap_ratio(items, scores, cut, positive)
        fair, above, tied = tie_fair(items, scores, cut, positive)
        lose, gain = counterfactual(items, scores, cut, positive)
        accuracy = sum(1 for i, (label, _) in items.items()
                       if (scores[i] >= 0.5) == (label == positive)) / len(items)
        rows.append({"pair": pair, "engine": engine, "variant": variant, "cut": cut,
                     "n_women_positive": n_women, "n_men_positive": n_men,
                     "accuracy": round(accuracy, 4),
                     "women_shortlist_rate": round(women_rate, 4), "men_shortlist_rate": round(men_rate, 4),
                     "four_fifths_ratio": round(ratio, 4), "ratio_ci": [round(low, 4), round(high, 4)],
                     "tie_fair_ratio": round(fair, 4), "above_cut": above, "tied_at_cut": tied,
                     # A real woman's twin is read as a man, a real man's as a woman.
                     "women_who_lose_place_read_as_men": lose["female"],
                     "men_who_lose_place_read_as_women": lose["male"],
                     "women_who_gain_place_read_as_men": gain["female"],
                     "men_who_gain_place_read_as_women": gain["male"]})
        print(f"{pair:20s} {engine:5s} {variant:13s} top {cut:4d}: ratio {ratio:.3f} [{low:.3f},{high:.3f}] "
              f"tie-fair {fair:.3f} (tied {tied}) | women in only as men {gain['female']}, "
              f"men out as women {lose['male']} | accuracy {accuracy:.3f}")


def main():
    rows = []
    for pair, positive in PAIRS.items():
        folder = Path("fixtures/bios_pairs") / pair
        items = load_items(folder)
        for engine, filename in ENGINES.items():
            scores = load_scores(folder / filename, positive)
            score(items, scores, positive, engine, pair, "engine_alone", rows)
            # Twin averaging: the engine alone, scored on the bio and on its pronoun-swapped
            # twin, and the two probabilities averaged. Zero pronoun flips by construction.
            averaged = {**scores, **{i: (scores[i] + scores[f"{i}-swapped"]) / 2 for i in items}}
            score(items, averaged, positive, engine, pair, "twin_averaged", rows)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
