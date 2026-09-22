#!/usr/bin/env python
"""Score the second race attempt (full names, four groups) from recorded answers.

    python scripts/run_bios_race2.py --engine laya --sample all
    python scripts/run_bios_race2.py --engine laya --sample 500
    python scripts/run_bios_race2.py --engine jev --sample 500

Reads ``fixtures/bios/race2_versions.jsonl`` (built by ``scripts/build_bios_race2_fixtures.py``)
and the engine's answer fixture (``fixtures/bios/answers-race2-laya.jsonl.gz`` for Laya,
``fixtures/bios/answers-race2.jsonl.gz`` for Jev, the latter covering only the 500-bio
subsample), groups each bio's answers into ``{group: [Verdict x4]}``, and appends one row to
``studies/bios_race2.jsonl`` via ``scripts/bios_race2.score_arm``.

``--sample 500`` restricts the bios scored to ``fixtures/bios/race2_jev_subsample.txt``, so
Laya can be compared to Jev on identical bios; ``--sample all`` (Laya only -- Jev never
answered the rest) scores every eligible bio. ``--excluded`` records how many of the 2,000
held-out bios had no insertion point at all (32, reported by the fixtures build; passed
through here rather than recomputed so this script needs no dependency on ``items.jsonl``).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, write_rows  # noqa: E402
from bios_race2 import score_arm  # noqa: E402

ANSWER_FILES = {"jev": "answers-race2.jsonl.gz", "laya": "answers-race2-laya.jsonl.gz"}
DEFAULT_EXCLUDED = 32  # see scripts/build_bios_race2_fixtures.py's own report


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(fixtures: Path, engine: str, sample: str, out: Path, excluded: int,
         subsample_path: Path) -> None:
    versions = {row["id"]: row for row in load_jsonl(fixtures / "race2_versions.jsonl")}
    answers_path = fixtures / ANSWER_FILES[engine]
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} does not exist -- see this script's docstring")
    question = "Occupation"
    answers = {row["id"]: row["answers"][question] for row in load_jsonl(answers_path)}

    restrict_ids = None
    if sample == "500":
        restrict_ids = {line.strip() for line in subsample_path.read_text().splitlines()
                        if line.strip()}
    elif sample != "all":
        raise SystemExit(f"--sample must be 'all' or '500', got {sample!r}")

    by_bio: dict = {}
    missing = []
    for version_id, row in versions.items():
        meta = row["metadata"]
        source_id = meta["source_id"]
        if restrict_ids is not None and source_id not in restrict_ids:
            continue
        if version_id not in answers:
            missing.append(version_id)
            continue
        a = answers[version_id]
        verdict = Verdict(source_id, a["choice"], a["probabilities"]["surgeon"],
                          meta["reference_label"], meta["gender"])
        by_bio.setdefault(source_id, {}).setdefault(meta["group"], [None, None, None, None])
        by_bio[source_id][meta["group"]][meta["k"] - 1] = verdict

    if missing:
        raise SystemExit(f"{len(missing)} versions have no {engine} answer yet, e.g. "
                          f"{missing[:3]}")

    # Every bio must have all 4 groups x 4 names before it can be scored.
    incomplete = [bio for bio, groups in by_bio.items()
                 if set(groups) != {"white", "black", "hispanic", "asian"}
                 or any(None in names for names in groups.values())]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} bios have incomplete groups, e.g. "
                         f"{incomplete[:3]}")

    metrics = score_arm(engine=engine, sample=sample, by_bio=by_bio, excluded=excluded)
    row = metrics.as_row()
    print(json.dumps(row, indent=2))
    write_rows([row], out)
    print(f"appended to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/bios"))
    parser.add_argument("--engine", choices=list(ANSWER_FILES), required=True)
    parser.add_argument("--sample", choices=("all", "500"), required=True)
    parser.add_argument("--out", type=Path, default=Path("studies/bios_race2.jsonl"))
    parser.add_argument("--excluded", type=int, default=DEFAULT_EXCLUDED)
    parser.add_argument("--subsample", type=Path,
                        default=Path("fixtures/bios/race2_jev_subsample.txt"))
    args = parser.parse_args()
    main(args.fixtures, args.engine, args.sample, args.out, args.excluded, args.subsample)
