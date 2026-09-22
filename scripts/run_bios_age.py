#!/usr/bin/env python
"""Score the age-insertion counterfactual study (both engines, engine-alone, no fitted head)
from recorded answers.

    python scripts/run_bios_age.py --engine jev
    python scripts/run_bios_age.py --engine laya

Reads ``fixtures/bios/age_versions.jsonl`` (built by ``scripts/build_bios_age_fixtures.py``)
and the engine's answer fixture (``fixtures/bios/answers-age.jsonl.gz`` for Jev,
``fixtures/bios/answers-age-laya.jsonl.gz`` for Laya), builds a ``Verdict`` for each of a
bio's four aged versions, and appends one row to ``studies/bios_age.jsonl`` via
``scripts/bios_age.score_arm``. ``--excluded`` records how many of the 2,000 held-out bios were
ineligible (no subject pronoun, a year before 2000, or a duration of 10+ years) and so could
not carry the age counterfactual (769, reported by the fixtures build; passed through here
rather than recomputed so this script needs no dependency on ``items.jsonl``).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, write_rows  # noqa: E402
from bios_age import score_arm  # noqa: E402

ANSWER_FILES = {"jev": "answers-age.jsonl.gz", "laya": "answers-age-laya.jsonl.gz"}
DEFAULT_EXCLUDED = 769  # see scripts/build_bios_age_fixtures.py's own report
AGES = (34, 35, 61, 62)


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(fixtures: Path, engine: str, out: Path, excluded: int) -> None:
    versions = {row["id"]: row for row in load_jsonl(fixtures / "age_versions.jsonl")}
    answers_path = fixtures / ANSWER_FILES[engine]
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} does not exist -- see this script's docstring")
    question = "Occupation"
    answers = {row["id"]: row["answers"][question] for row in load_jsonl(answers_path)}

    def verdict(source_id: str, version_id: str) -> Verdict:
        """A version's answer, keyed by the bio's *source* id (shared across its four aged
        versions) so ``score_arm`` can pair 34/35/61/62 for the same bio."""
        a = answers[version_id]
        meta = versions[version_id]["metadata"]
        return Verdict(source_id, a["choice"], a["probabilities"]["surgeon"],
                       meta["reference_label"], meta["gender"])

    by_source: dict = {}
    for version_id, row in versions.items():
        by_source.setdefault(row["metadata"]["source_id"], {})[row["metadata"]["age"]] = \
            version_id

    missing = [vid for vid in versions if vid not in answers]
    if missing:
        raise SystemExit(f"{len(missing)} versions have no {engine} answer yet, e.g. "
                          f"{missing[:3]}")

    v34 = [verdict(source_id, ids[34]) for source_id, ids in by_source.items()]
    v35 = {source_id: verdict(source_id, ids[35]) for source_id, ids in by_source.items()}
    v61 = {source_id: verdict(source_id, ids[61]) for source_id, ids in by_source.items()}
    v62 = {source_id: verdict(source_id, ids[62]) for source_id, ids in by_source.items()}

    metrics = score_arm(engine=engine, v34=v34, v35=v35, v61=v61, v62=v62, excluded=excluded)
    row = metrics.as_row()
    print(json.dumps(row, indent=2))
    write_rows([row], out)
    print(f"appended to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/bios"))
    parser.add_argument("--engine", choices=list(ANSWER_FILES), required=True)
    parser.add_argument("--out", type=Path, default=Path("studies/bios_age.jsonl"))
    parser.add_argument("--excluded", type=int, default=DEFAULT_EXCLUDED)
    args = parser.parse_args()
    main(args.fixtures, args.engine, args.out, args.excluded)
