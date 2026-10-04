#!/usr/bin/env python
"""Turn four initial JEVPA analyst replies into auditable candidate scorecards.

Only question keys are mechanically disambiguated when analysts reuse a key with
different wording. The question types, criteria, and instructions remain exactly
what the analyst proposed. The combined scorecard exists only to batch Jev's
answers; every candidate is fit separately later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.proposal import ElementAdd, Proposal, apply_proposal, parse_proposal
from jev_flywheel.scorecard import Scorecard


def build(proposals_path: Path, base_path: Path, out_dir: Path, *, score_name: str = "Sentiment") -> dict:
    """Write one candidate per initial reply and their question-union scorecard."""
    base = Scorecard.from_yaml(Path(base_path).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(proposals_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 4:
        raise ValueError("the protocol requires exactly four initial proposal records")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_additions, candidates, seen_keys = [], [], set()
    for row in sorted(rows, key=lambda item: item["call_index"]):
        proposal = parse_proposal(row["analyst_reply"])
        if proposal.retire or proposal.reword or len(proposal.add) != 1:
            raise ValueError(f"call {row['call_index']} must add exactly one initial question")
        added = proposal.add[0]
        key = added.key
        if key in seen_keys:
            key = f"{key}_{row['call_index']}"
        seen_keys.add(key)
        canonical = ElementAdd(key, added.question_type, added.instructions, added.criteria,
                               added.features)
        candidate = apply_proposal(base, score_name, Proposal(root_cause=proposal.root_cause,
                                                               add=[canonical]))
        candidate_path = out_dir / f"initial-{row['call_index']}.yaml"
        candidate_path.write_text(candidate.to_yaml(), encoding="utf-8")
        candidates.append({"call_index": row["call_index"], "candidate": str(candidate_path),
                           "original_key": added.key, "question_key": key,
                           "instructions": added.instructions})
        all_additions.append(canonical)
    batch = apply_proposal(base, score_name, Proposal(add=all_additions))
    batch_path = out_dir / "initial-batch.yaml"
    batch_path.write_text(batch.to_yaml(), encoding="utf-8")
    result = {"candidates": candidates, "batch_scorecard": str(batch_path),
              "question_count": len(batch.questions())}
    (out_dir / "initial-candidates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("fixtures/scorecards/v1.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.proposals, args.base, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
