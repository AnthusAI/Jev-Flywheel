#!/usr/bin/env python
"""Collect JEVPA's four matched initial analyst proposals without calling Jev.

Each call starts from the identical v1 scorecard and the same recorded 140
discovery labels. ``run_steering`` is intentionally called with
``allow_spend=False``: the procedure can ask the analyst and validate its edit,
but it stops before Jev would answer a new question. The JSONL output is the
audit record used to select and price the shared first answer batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.items import FeedbackItem, JsonlStore
from jev_flywheel.steer import DEFAULT_MODEL, SteerOutcome, run_steering
from jev_flywheel.workspace import Workspace


def seed_workspace(root: Path, recording: Path) -> tuple[Workspace, str]:
    """Create a disposable workspace containing only the registered discovery labels."""
    workspace = Workspace.init(root, PACKAGED_FIXTURES)
    feedback = JsonlStore(Path(recording) / "feedback.jsonl", FeedbackItem).all()
    score_name = workspace.scorecard().scores[0].name
    for entry in feedback:
        if entry.score_name == score_name:
            workspace.add_feedback(entry)
    if workspace.n_labeled(score_name) != 140:
        raise ValueError("the live protocol requires exactly 140 usable discovery labels")
    return workspace, score_name


def collect(
    out: Path, workspace_root: Path, recording: Path, *, model: str = DEFAULT_MODEL,
    n_calls: int = 4, run_round: Callable[..., SteerOutcome] = run_steering,
) -> list[dict]:
    """Collect missing proposal slots, never authorizing Jev spend."""
    if n_calls < 1:
        raise ValueError("n_calls must be positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    records = []
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add(record["call_index"])
                records.append(record)
    for index in range(1, n_calls + 1):
        if index in done:
            continue
        run_root = workspace_root / f"shared-initial-{index}"
        if run_root.exists():
            raise ValueError(f"{run_root} already exists without a completed record; refuse to overwrite it")
        workspace, score_name = seed_workspace(run_root, recording)
        outcome = run_round(
            workspace, score_name, model=model, allow_spend=False,
            # The no-spend host returns ``needs_spend`` before a proposed scorecard could
            # be applied. No scripted approval can authorize a model or Jev call here.
            max_auto_requests=10_000,
        )
        record = {
            "phase": "shared_initial",
            "call_index": index,
            "model": model,
            "allow_spend": False,
            "discovery_labels": workspace.n_labeled(score_name),
            "decision": outcome.decision,
            "detail": outcome.detail,
            "analyst_reply": outcome.analyst_reply,
        }
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        records.append(record)
    return sorted(records, key=lambda row: row["call_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("var/jevpa_live/initial_proposals.jsonl"))
    parser.add_argument("--workspace-root", type=Path, default=Path("var/jevpa_live/workspaces"))
    parser.add_argument("--recording", type=Path,
                        default=Path("fixtures/recordings/simulated-labeler"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    records = collect(args.out, args.workspace_root, args.recording, model=args.model)
    print(json.dumps({"out": str(args.out), "calls": len(records),
                      "decisions": [row["decision"] for row in records]}, sort_keys=True))


if __name__ == "__main__":
    main()
