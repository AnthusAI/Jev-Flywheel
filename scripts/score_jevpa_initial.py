#!/usr/bin/env python
"""Fit and select JEVPA's shared initial candidates without exposing labels to an analyst."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.answers import AnswerCache
from jev_flywheel.items import FeedbackItem, JsonlStore
from jev_flywheel.jevpa import Candidate, has_distinct_slice_specialists, pareto_pool
from jev_flywheel.scorecard import Scorecard
from scripts.run_jevpa_offline import _fit_candidate, _jsonl


def score_initial(candidates_path: Path, items_path: Path, feedback_path: Path,
                  split_path: Path, cache_path: Path, out_dir: Path, *, seed: int = 20260925) -> dict:
    """Fit only discovery labels and calculate selection-only candidate summaries."""
    candidate_manifest = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    rows = list(_jsonl(Path(items_path)))
    items = {row["id"]: row for row in rows}
    feedback = list(JsonlStore(Path(feedback_path), FeedbackItem))
    discovery_ids = set(splits["discovery_ids"])
    discovery = [entry for entry in feedback if entry.item_id in discovery_ids and entry.label is not None]
    if len(discovery) != 140:
        raise ValueError("initial candidates must fit exactly the 140 registered discovery labels")
    selection_ids = list(splits["selection_ids"])
    cache = AnswerCache(cache_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    records = []
    for entry in candidate_manifest["candidates"]:
        card = Scorecard.from_yaml(Path(entry["candidate"]).read_text(encoding="utf-8"))
        candidate, record = _fit_candidate(f"initial:{entry['call_index']}", card, discovery,
                                           cache, selection_ids, items, seed)
        fitted_path = out_dir / f"initial-{entry['call_index']}-fitted.yaml"
        fitted_path.write_text(candidate.scorecard.to_yaml(), encoding="utf-8")
        record["fitted_scorecard"] = str(fitted_path)
        candidates.append(candidate)
        records.append(record)
    frontier = pareto_pool(candidates, max_size=4, min_slice_effective_n=50)
    winner = min(candidates, key=lambda item: (item.overall_brier, item.candidate_id))
    report = {
        "discovery_labels": len(discovery), "selection_items": len(selection_ids),
        "candidates": records, "overall_winner": winner.candidate_id,
        "pareto_pool": [item.candidate_id for item in frontier],
        "distinct_slice_specialists": has_distinct_slice_specialists(
            candidates, min_slice_effective_n=50),
    }
    (out_dir / "initial_selection.json").write_text(json.dumps(report, indent=2) + "\n",
                                                       encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=Path("fixtures/items.jsonl"))
    parser.add_argument("--feedback", type=Path,
                        default=Path("fixtures/recordings/simulated-labeler/feedback.jsonl"))
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_initial(args.candidates, args.items, args.feedback, args.splits,
                                   args.cache, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
