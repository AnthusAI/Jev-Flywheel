#!/usr/bin/env python
"""Freeze the fresh sentiment splits required by the JEVPA live protocol.

This command is deliberately offline: it reads the fixture and the recorded
feedback, excludes every item used by the prior pilot, and writes the immutable
ID manifest that must exist before any analyst or Jev call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_flywheel.items import FeedbackItem, JsonlStore
from jev_flywheel.jevpa import fresh_split_ids
from scripts.run_jevpa_offline import _jsonl, partition_pool_ids


def split_manifest(fixtures: Path, recording: Path, *, seed: int = 20260925) -> dict:
    """Create the registered 300/500 live split after pilot and discovery exclusions."""
    rows = list(_jsonl(Path(fixtures) / "items.jsonl"))
    feedback = list(JsonlStore(Path(recording) / "feedback.jsonl", FeedbackItem))
    discovery = {entry.item_id for entry in feedback if entry.label is not None}
    pilot_selection, pilot_test = partition_pool_ids(
        rows, discovery, seed=20260924, selection_size=500, test_size=600)
    pool_ids = [row["id"] for row in rows if row.get("metadata", {}).get("split") == "pool"]
    selection, test = fresh_split_ids(
        pool_ids, excluded_ids=discovery | set(pilot_selection) | set(pilot_test),
        seed=seed, selection_size=300, test_size=500)
    payload = {
        "experiment": "JEVPA live sentiment comparison",
        "split_seed": seed,
        "discovery_ids": sorted(discovery),
        "excluded_prior_pilot": {"selection_ids": pilot_selection, "test_ids": pilot_test},
        "selection_ids": selection,
        "test_ids": test,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    parser.add_argument("--recording", type=Path,
                        default=Path("fixtures/recordings/simulated-labeler"))
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = split_manifest(args.fixtures, args.recording, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": result["sha256"],
                      "selection_items": len(result["selection_ids"]),
                      "test_items": len(result["test_ids"])}, sort_keys=True))


if __name__ == "__main__":
    main()
