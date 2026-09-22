#!/usr/bin/env python
"""Score the engine-alone arms (J0, L0) of the gender-swap study from recorded answers.

    python scripts/run_bios_arms.py --arm L0
    python scripts/run_bios_arms.py --arm J0     # needs fixtures/bios/answers.jsonl.gz

Reads ``fixtures/bios/items.jsonl`` and the arm's answer fixture, builds a ``Verdict`` for every
held-out bio and its counterfactual twin, and appends one row to ``studies/bios_gender.jsonl``
via ``scripts/bios_gender.score_arm``. This is deliberately the only script that reads the answer
fixtures directly for J0/L0 -- later arms (J1/J2/L1/L2/LF) build a workspace over the same
fixtures and go through the flywheel's own fit/steer path instead, but score with the same
``bios_gender`` functions.

J0 needs ``fixtures/bios/answers.jsonl.gz`` (Jev's answers), which this repo does not ship yet:
producing it needs ``TYPESAFE_API_KEY`` (see ``.env.example``), and was not available when the
corpus and L0 were built (``studies/PREREGISTERED.md``'s deviations note says so).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, score_arm, write_rows  # noqa: E402

ANSWER_FILES = {"J0": "answers.jsonl.gz", "L0": "answers-laya.jsonl.gz"}
ENGINES = {"J0": "jev", "L0": "laya"}


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(fixtures: Path, arm: str, out: Path, redacted: bool = True) -> None:
    items = {row["id"]: row for row in load_jsonl(fixtures / "items.jsonl")}
    answers_path = fixtures / ANSWER_FILES[arm]
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} does not exist -- see this script's docstring")
    question = "Occupation"
    answers = {row["id"]: row["answers"][question] for row in load_jsonl(answers_path)}

    def verdict(item_id: str) -> Verdict:
        a = answers[item_id]
        meta = items[item_id]["metadata"]
        return Verdict(item_id, a["choice"], a["probabilities"]["surgeon"],
                       meta["reference_label"], meta["gender"])

    test_ids = [i for i, r in items.items() if r["metadata"]["split"] == "test"]
    twin_of = {r["metadata"]["counterfactual_of"]: i for i, r in items.items()
               if r["metadata"]["split"] == "counterfactual"}
    missing = [i for i in test_ids if i not in answers] + [t for t in twin_of.values() if t not in answers]
    if missing:
        raise SystemExit(f"{len(missing)} items have no {arm} answer yet, e.g. {missing[:3]}")

    verdicts = [verdict(i) for i in test_ids]
    twins = {orig: verdict(twin_id) for orig, twin_id in twin_of.items()}
    metrics = score_arm(arm=arm, engine=ENGINES[arm], verdicts=verdicts, twins=twins,
                        redacted=redacted)
    row = metrics.as_row()
    print(json.dumps(row, indent=2))
    write_rows([row], out)
    print(f"appended to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/bios"))
    parser.add_argument("--arm", choices=list(ANSWER_FILES), required=True)
    parser.add_argument("--out", type=Path, default=Path("studies/bios_gender.jsonl"))
    parser.add_argument("--not-redacted", action="store_true",
                        help="tag the row redacted:false (for replaying the pre-redaction "
                             "fixtures kept under var/backup_pre_redaction)")
    args = parser.parse_args()
    main(args.fixtures, args.arm, args.out, redacted=not args.not_redacted)
