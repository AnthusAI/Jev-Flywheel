#!/usr/bin/env python
"""Create the offline Civil Comments JEVPA workspace from the sealed corpus.

It stores labels only for the registered discovery rows. Selection and test labels are
kept in the external sealed evaluation file, never copied into this workspace.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.items import FeedbackItem, JsonlStore, LABEL_SOURCE_VETTED
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.workspace import Workspace


SCORE_NAME = "Toxicity"
V1 = """name: Toxicity
version: 1
scores:
  - name: Toxicity
    key: toxicity
    question_type: choice
    instructions: Should this comment be removed under a toxicity moderation policy?
    criteria: {remove: null, keep: null}
    decision:
      model: linear_threshold
      classes: [remove, keep]
      positive_class: remove
      threshold: 0.0
      features: [self.holistic.clr.remove]
      parameters:
        weights: {intercept: 0.0, self.holistic.clr.remove: 2.0}
"""


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prepare(corpus: Path, analyst_discovery: Path, root: Path, *,
            expected_discovery_size: int = 140) -> Workspace:
    """Build a fresh workspace and seed exactly the trusted discovery feedback."""
    root = Path(root)
    if root.exists():
        raise ValueError(f"workspace already exists at {root}; refusing to overwrite it")
    all_rows = _rows(corpus)
    labels = _rows(analyst_discovery)
    by_id = {row["id"]: row for row in all_rows}
    if len(labels) != expected_discovery_size and len(all_rows) > 3:
        raise ValueError(f"Civil Comments protocol requires exactly {expected_discovery_size} discovery labels")
    if any(row["id"] not in by_id for row in labels):
        raise ValueError("a discovery label names an item absent from the corpus")
    if any(by_id[row["id"]]["metadata"].get("split") != "discovery" for row in labels):
        raise ValueError("only discovery items may carry analyst labels")
    root.mkdir(parents=True)
    shutil.copyfile(corpus, root / "items.jsonl")
    (root / "workspace.json").write_text(json.dumps({"engine": "jev", "study": "civil-comments-jevpa"}) + "\n")
    workspace = Workspace(root)
    workspace.commit_scorecard(Scorecard.from_yaml(V1), kind="seed",
                               provenance={"study": "Civil Comments JEVPA"})
    feedback = []
    for index, row in enumerate(labels):
        label = "remove" if int(row["metadata"]["reference_label"]) else "keep"
        feedback.append(FeedbackItem(
            id=f"civil-discovery-{index:03d}", item_id=row["id"], score_name=SCORE_NAME,
            final_answer_value=label, label_source=LABEL_SOURCE_VETTED,
            metadata={"propensity": 1.0, "source": "Civil Comments toxicity >= 0.5"},
        ))
    JsonlStore(workspace.feedback_path, FeedbackItem).append_all(feedback)
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-discovery-size", type=int, default=140)
    args = parser.parse_args()
    workspace = prepare(args.corpus, args.discovery, args.out,
                        expected_discovery_size=args.expected_discovery_size)
    print(json.dumps({"workspace": str(workspace.root), "labels": workspace.n_labeled(SCORE_NAME)}))


if __name__ == "__main__":
    main()
