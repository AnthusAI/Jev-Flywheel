#!/usr/bin/env python
"""Recompute studies/bios_nurse_shortlist.jsonl with the tie diagnostic, for arms already run.

Read-only: reloads each arm's already-built workspace under var/bios_nurse_loop/<arm>-seed<n>/var
(J0 reads straight from fixtures/bios_nurse/answers.jsonl.gz, as run_engine_alone does) and
recomputes scores from the cached answers already on disk -- no new Jev requests, no Laya calls,
no GPU. Overwrites studies/bios_nurse_shortlist.jsonl from scratch so it holds exactly one row
per (arm, seed, cut), all carrying the tie fields added to
``scripts/bios_nurse_shortlist.shortlist_metrics``.

    python scripts/bios_nurse_tie_recompute.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_nurse_shortlist import shortlist_metrics  # noqa: E402

from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

FIXTURES = Path("fixtures/bios_nurse")
SCRATCH = Path("var/bios_nurse_loop")
OUT = Path("studies/bios_nurse_shortlist.jsonl")
POSITIVE = "physician"
SCORE_NAME = "Occupation"

ARMS_TO_REPLAY = [("J1", [1, 2, 3]), ("J2", [1, 2, 3])]


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_held_out_items() -> Dict[str, dict]:
    return {row["id"]: row for row in load_jsonl(FIXTURES / "items.jsonl")
            if row["metadata"]["split"] in ("test", "counterfactual")}


def held_out_split():
    items = load_held_out_items()
    test_ids = [i for i, r in items.items() if r["metadata"]["split"] == "test"]
    twin_of = {r["metadata"]["counterfactual_of"]: i for i, r in items.items()
               if r["metadata"]["split"] == "counterfactual"}
    return test_ids, twin_of


def run_shortlist_rows(arm: str, engine: str, seed, scores_by_id: Dict[str, float]) -> list:
    items_meta = load_held_out_items()
    test_ids, twin_of = held_out_split()
    items = {i: (items_meta[i]["metadata"]["reference_label"], items_meta[i]["metadata"]["gender"])
             for i in test_ids}
    scores = {i: scores_by_id[i] for i in test_ids}
    twin_scores = {orig: scores_by_id[twin_id] for orig, twin_id in twin_of.items()}
    rows = shortlist_metrics(items, scores, twin_scores, positive_label=POSITIVE)
    for row in rows:
        row.update({"arm": arm, "engine": engine, "seed": seed})
    return rows


def j0_scores() -> Dict[str, float]:
    answers = {row["id"]: row["answers"]["Occupation"]
               for row in load_jsonl(FIXTURES / "answers.jsonl.gz")}
    return {i: a["probabilities"][POSITIVE] for i, a in answers.items()}


def arm_scores(arm: str, seed: int) -> Dict[str, float]:
    ws = Workspace(SCRATCH / f"{arm}-seed{seed}" / "var")
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    test_items = ws.split("test")
    twin_items = ws.split("counterfactual")
    scores_by_id: Dict[str, float] = {}
    for item in test_items + twin_items:
        answers = ws.cache.partial_answers_for(item.id, questions)
        result = predict(score, answers)
        confidence = result.confidence if result.confidence is not None else 0.5
        scores_by_id[item.id] = confidence if result.value == POSITIVE else 1.0 - confidence
    return scores_by_id


def main() -> None:
    all_rows = []
    print("J0:")
    all_rows.extend(run_shortlist_rows("J0", "jev", None, j0_scores()))

    for arm, seeds in ARMS_TO_REPLAY:
        for seed in seeds:
            ws_root = SCRATCH / f"{arm}-seed{seed}" / "var"
            if not ws_root.exists():
                print(f"  {arm} seed {seed}: no workspace at {ws_root}, skipping")
                continue
            print(f"{arm} seed {seed}:")
            rows = run_shortlist_rows(arm, "jev", seed, arm_scores(arm, seed))
            all_rows.extend(rows)
            for r in rows:
                print(f"  top{r['cut']}: ratio={r['four_fifths_ratio']} "
                      f"tie_fair_ratio={r['tie_fair_four_fifths_ratio']} "
                      f"n_tied_at_cut={r['n_tied_at_cut']}")

    with OUT.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
