#!/usr/bin/env python
"""Score the "does the gender result hold on other decisions?" study from recorded answers.

    python scripts/run_bios_pairs.py --pair nurse_physician --engine jev
    python scripts/run_bios_pairs.py --pair nurse_physician --engine laya
    python scripts/run_bios_pairs.py --pair paralegal_attorney --engine jev
    python scripts/run_bios_pairs.py --pair paralegal_attorney --engine laya
    python scripts/run_bios_pairs.py --pair teacher_professor --engine jev
    python scripts/run_bios_pairs.py --pair teacher_professor --engine laya
    python scripts/run_bios_pairs.py --copy-surgeon-physician   # 4th row, from bios_gender.jsonl

Reads ``fixtures/bios_pairs/<pair>/items.jsonl`` and the engine's answer fixture
(``answers.jsonl.gz`` for Jev, ``answers-laya.jsonl.gz`` for Laya), builds a ``Verdict`` for
every item and its counterfactual twin (every item here is held out -- no pool), and appends one
row to ``studies/bios_pairs.jsonl`` via ``scripts/bios_pairs.score_pair``.

``--copy-surgeon-physician`` appends the existing surgeon/physician J0/L0 rows from
``studies/bios_gender.jsonl`` (the most recent ``redacted: true`` row per engine), reshaped into
the same schema and tagged ``source: "bios_gender"``, so the combined four-pair table can be
built from ``studies/bios_pairs.jsonl`` alone.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, write_rows  # noqa: E402
from bios_pairs import PAIR_INFO, score_pair  # noqa: E402

ANSWER_FILES = {"jev": "answers.jsonl.gz", "laya": "answers-laya.jsonl.gz"}
ENGINE_ARM = {"jev": "J0", "laya": "L0"}


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_from_fixtures(pair: str, engine: str, fixtures_root: Path, out: Path) -> None:
    info = PAIR_INFO[pair]
    less_female = info["less_female"]
    fixtures = fixtures_root / pair
    items = {row["id"]: row for row in load_jsonl(fixtures / "items.jsonl")}
    answers_path = fixtures / ANSWER_FILES[engine]
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} does not exist -- see this script's docstring")
    question = "Occupation"
    answers = {row["id"]: row["answers"][question] for row in load_jsonl(answers_path)}

    def verdict(item_id: str) -> Verdict:
        a = answers[item_id]
        meta = items[item_id]["metadata"]
        return Verdict(item_id, a["choice"], a["probabilities"][less_female],
                       meta["reference_label"], meta["gender"])

    test_ids = [i for i, r in items.items() if r["metadata"]["split"] == "test"]
    twin_of = {r["metadata"]["counterfactual_of"]: i for i, r in items.items()
               if r["metadata"]["split"] == "counterfactual"}
    missing = ([i for i in test_ids if i not in answers]
               + [t for t in twin_of.values() if t not in answers])
    if missing:
        raise SystemExit(f"{len(missing)} items have no {engine} answer yet, e.g. {missing[:3]}")

    verdicts = [verdict(i) for i in test_ids]
    twins = {orig: verdict(twin_id) for orig, twin_id in twin_of.items()}
    metrics = score_pair(pair=pair, engine=engine, verdicts=verdicts, twins=twins)
    row = metrics.as_row()
    print(json.dumps(row, indent=2))
    write_rows([row], out)
    print(f"appended to {out}")


def copy_surgeon_physician(gender_jsonl: Path, out: Path) -> None:
    """Append the existing surgeon/physician J0/L0 rows, reshaped to the bios_pairs schema."""
    rows = list(load_jsonl(gender_jsonl))
    latest = {}
    for row in rows:
        if row.get("redacted") is True and row["arm"] in ("J0", "L0"):
            latest[row["arm"]] = row  # later rows overwrite earlier ones -> "latest"
    if not latest:
        raise SystemExit(f"no redacted J0/L0 rows found in {gender_jsonl}")

    info = PAIR_INFO["surgeon_physician"]
    out_rows = []
    for arm, engine in (("J0", "jev"), ("L0", "laya")):
        if arm not in latest:
            print(f"  skipping {arm}: no redacted row in {gender_jsonl}")
            continue
        source = latest[arm]
        out_rows.append({
            "pair": "surgeon_physician", "engine": engine,
            "less_female": info["less_female"], "more_female": info["more_female"],
            "gap_points": info["gap_points"], "n": source["n"],
            "accuracy": source["accuracy"],
            "counterfactual_flip_rate": source["counterfactual_flip_rate"],
            "flip_rate_ci": None,  # bios_gender.jsonl does not record a bootstrap CI
            "mean_abs_delta_p": source["mean_abs_delta_p"],
            "flip_toward_more_female_share": source["flip_toward_physician_share"],
            "recall_gap_less_female_women_minus_men": source["tpr_gap_surgeon_women_minus_men"],
            "ece": source["ece"],
            "source": "bios_gender",
        })
    write_rows(out_rows, out)
    for row in out_rows:
        print(json.dumps(row, indent=2))
    print(f"appended {len(out_rows)} row(s) to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures-root", type=Path, default=Path("fixtures/bios_pairs"))
    parser.add_argument("--pair", choices=[p for p in PAIR_INFO if p != "surgeon_physician"])
    parser.add_argument("--engine", choices=list(ANSWER_FILES))
    parser.add_argument("--out", type=Path, default=Path("studies/bios_pairs.jsonl"))
    parser.add_argument("--copy-surgeon-physician", action="store_true",
                        help="append the surgeon/physician J0/L0 rows from bios_gender.jsonl")
    parser.add_argument("--gender-jsonl", type=Path, default=Path("studies/bios_gender.jsonl"))
    args = parser.parse_args()

    if args.copy_surgeon_physician:
        copy_surgeon_physician(args.gender_jsonl, args.out)
        return
    if not args.pair or not args.engine:
        raise SystemExit("--pair and --engine are required unless --copy-surgeon-physician")
    score_from_fixtures(args.pair, args.engine, args.fixtures_root, args.out)


if __name__ == "__main__":
    main()
