#!/usr/bin/env python
"""Collect the two matched greedy and two Pareto JEVPA branch proposals without Jev spend."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.items import FeedbackItem, JsonlStore
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.steer import DEFAULT_MODEL, SteerOutcome, run_steering
from jev_flywheel.workspace import Workspace


def branch_workspace(root: Path, recording: Path, parent_scorecard: Path,
                     live_cache: Path) -> tuple[Workspace, str]:
    """Create one branch with the candidate parent and only discovery labels."""
    workspace = Workspace.init(root, PACKAGED_FIXTURES)
    shutil.copyfile(live_cache, workspace.answers_path)
    workspace._cache = None
    score_name = workspace.scorecard().scores[0].name
    parent = Scorecard.from_yaml(Path(parent_scorecard).read_text(encoding="utf-8"))
    workspace.commit_scorecard(parent, kind="jevpa_branch_parent")
    for entry in JsonlStore(Path(recording) / "feedback.jsonl", FeedbackItem).all():
        if entry.score_name == score_name:
            workspace.add_feedback(entry)
    if workspace.n_labeled(score_name) != 140:
        raise ValueError("the live protocol requires exactly 140 usable discovery labels")
    return workspace, score_name


def collect(out: Path, workspace_root: Path, recording: Path, live_cache: Path,
            greedy_parent: Path, pareto_parent: Path, *, model: str = DEFAULT_MODEL,
            run_round: Callable[..., SteerOutcome] = run_steering) -> list[dict]:
    """Run the four registered expansion calls, stopping before any Jev top-up."""
    out.parent.mkdir(parents=True, exist_ok=True)
    schedule = [("greedy", index, greedy_parent) for index in (1, 2)] + [
        ("pareto", index, pareto_parent) for index in (1, 2)]
    records = []
    for arm, index, parent in schedule:
        root = workspace_root / f"{arm}-{index}"
        if root.exists():
            raise ValueError(f"{root} already exists; branch calls are not rerun")
        workspace, score_name = branch_workspace(root, recording, parent, live_cache)
        outcome = run_round(workspace, score_name, model=model, allow_spend=False,
                            max_auto_requests=10_000)
        record = {"phase": arm, "call_index": index, "parent_scorecard": str(parent),
                  "model": model, "allow_spend": False,
                  "discovery_labels": workspace.n_labeled(score_name),
                  "decision": outcome.decision, "detail": outcome.detail,
                  "analyst_reply": outcome.analyst_reply}
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("var/jevpa_live/branch_proposals.jsonl"))
    parser.add_argument("--workspace-root", type=Path, default=Path("var/jevpa_live/branch_workspaces"))
    parser.add_argument("--recording", type=Path,
                        default=Path("fixtures/recordings/simulated-labeler"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--greedy-parent", type=Path, required=True)
    parser.add_argument("--pareto-parent", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    results = collect(args.out, args.workspace_root, args.recording, args.cache,
                      args.greedy_parent, args.pareto_parent, model=args.model)
    print(json.dumps({"out": str(args.out), "calls": len(results),
                      "decisions": [row["decision"] for row in results]}, sort_keys=True))


if __name__ == "__main__":
    main()
