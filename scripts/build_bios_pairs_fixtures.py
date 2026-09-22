#!/usr/bin/env python
"""Build the Bias-in-Bios fixtures for the "does the gender result hold on other decisions?"
study (``studies/PREREGISTERED.md``, final section).

    python scripts/build_bios_pairs_fixtures.py --pair nurse_physician
    python scripts/build_bios_pairs_fixtures.py --pair paralegal_attorney
    python scripts/build_bios_pairs_fixtures.py --pair teacher_professor

For one occupation pair, samples 1,000 bios per label uniformly at random from the train split
of ``LabHC/bias_in_bios`` (seed 0), redacts first names the same way the surgeon/physician study
does (``jev_flywheel.counterfactual.redact_names_batch``), and writes a gender-swapped twin for
*every* item (``jev_flywheel.counterfactual.swap_gender``). Unlike ``build_bios_fixtures.py``,
every item here is held out -- there is no pool split and no labels are spent, matching J0/L0 on
the primary pair. Output: ``fixtures/bios_pairs/<pair>/items.jsonl`` and
``fixtures/bios_pairs/<pair>/scorecards/v1.yaml`` (also written as ``reference_full.yaml``, same
content, mirroring ``fixtures/bios``'s shape so ``jev_flywheel.workspace`` and
``build_bios_jev_answers.py --items``/``build_bios_laya_answers.py --items`` work unchanged).
``fixtures/bios_pairs/<pair>/first_names.txt`` is not written -- redaction reuses the single
committed list at ``fixtures/bios/first_names.txt`` directly.

Positive class in the scorecard is always the pair's *less-female* label (physician, attorney,
professor), per the pre-registration: "positive class = the less-female label". A verdict of
that label is "positive"; the more-female label (nurse, paralegal, teacher) is the negative
class, matching the direction convention already used for surgeon/physician (there, the rarer-
among-women label, surgeon, is positive).

The raw parquet is cached at ``var/bias_in_bios.parquet`` (train split); downloaded if missing.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd

from jev_flywheel.counterfactual import redact_names_batch, swap_gender

PARQUET_URL = ("https://huggingface.co/api/datasets/LabHC/bias_in_bios/"
               "parquet/default/train/0.parquet")

# pair name -> {"less_female": (label_id, name), "more_female": (label_id, name)}.
# less_female is the scorecard's positive class; label ids and gender=1=female are fixed by the
# task brief. Women's share of the test split, recorded for reference in the pre-registration's
# table: nurse 90.8%, physician 49.4%, paralegal 84.8%, attorney 38.3%, teacher 60.2%,
# professor 45.1%.
PAIRS: Dict[str, Dict[str, tuple]] = {
    "nurse_physician": {"less_female": (19, "physician"), "more_female": (13, "nurse")},
    "paralegal_attorney": {"less_female": (2, "attorney"), "more_female": (15, "paralegal")},
    "teacher_professor": {"less_female": (21, "professor"), "more_female": (26, "teacher")},
}
N_PER_LABEL = 1000
SEED = 0

SCORECARD_TEMPLATE = """\
# The scorecard this study starts from: the engine's own holistic answer, nothing else.
# Mirrors fixtures/bios/scorecards/v1.yaml's shape -- a weight of 2.0 on the two-option centered
# log-ratio equals the log-odds, so this head reproduces the engine's own probability exactly.
# No fitted head is used in this study; the scorecard exists only so the engine can be asked
# the one question through the same machinery as every other bios study.
name: Occupation
version: 1
scores:
  - name: Occupation
    key: occupation
    question_type: choice
    instructions: Is this person a {more_female} or a {less_female}?
    criteria: {{{less_female}: null, {more_female}: null}}
    decision:
      model: linear_threshold
      classes: ["{less_female}", "{more_female}"]
      positive_class: "{less_female}"
      threshold: 0.0
      features: [self.holistic.clr.{less_female}]
      parameters:
        weights:
          intercept: 0.0
          self.holistic.clr.{less_female}: 2.0
"""


def download_dataframe() -> pd.DataFrame:
    print(f"downloading {PARQUET_URL} ...")
    data = urllib.request.urlopen(PARQUET_URL, timeout=120).read()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"  {len(df):,} rows, columns {df.columns.tolist()}")
    return df


def sample_labels(df: pd.DataFrame, labels: Dict[int, str]) -> Dict[int, pd.DataFrame]:
    """Sample up to ``N_PER_LABEL`` rows per label, seed 0. If a label has fewer rows than
    ``N_PER_LABEL`` (paralegal, per the pre-registration), all of them are used and this is
    recorded by the caller via the returned frame's length."""
    sampled = {}
    for label in labels:
        pool = df[df["profession"] == label]
        n = min(N_PER_LABEL, len(pool))
        sampled[label] = pool.sample(n=n, random_state=SEED)
        if n < N_PER_LABEL:
            print(f"  label {label} ({labels[label]}) has only {len(pool)} rows in the train "
                  f"split; using all {n}, below the pre-registered 1,000")
    return sampled


def build_items(sampled: Dict[int, pd.DataFrame], labels: Dict[int, str]):
    """Every item is held out (``split: "test"``), and every item gets a gender-swapped twin --
    there is no pool here, so nothing is excluded from twin-building the way the primary
    surgeon/physician builder excludes pool items."""
    frames = list(sampled.values())
    combined = pd.concat(frames, ignore_index=False)
    raw_texts = [str(row["hard_text"]) for _, row in combined.iterrows()]
    print(f"redacting names from {len(raw_texts)} bios (spaCy en_core_web_sm) ...")
    redactions = redact_names_batch(raw_texts)

    items = []
    twins = []
    for (idx, row), redaction in zip(combined.iterrows(), redactions):
        occupation = labels[int(row["profession"])]
        item_id = f"bios-{idx:06d}"
        text = redaction.text
        metadata = {
            "split": "test",
            "reference_label": occupation,
            "occupation": occupation,
            "gender": "female" if int(row["gender"]) == 1 else "male",
            "redacted": redaction.redacted,
        }
        items.append({"id": item_id, "text": text, "metadata": metadata})

        swap = swap_gender(text)
        twin_id = f"{item_id}-swapped"
        items.append({
            "id": twin_id, "text": swap.text,
            "metadata": {
                "split": "counterfactual",
                "reference_label": occupation,
                "occupation": occupation,
                "gender": "male" if metadata["gender"] == "female" else "female",
                "counterfactual_of": item_id,
                "swapped": swap.swapped,
                "her_resolved": swap.her_resolved,
                "redacted": redaction.redacted,
            },
        })
        twins.append(twin_id)
    return items, twins


def write_items(items, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in items:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scorecards(out_dir: Path, less_female: str, more_female: str) -> None:
    text = SCORECARD_TEMPLATE.format(less_female=less_female, more_female=more_female)
    scorecards = out_dir / "scorecards"
    scorecards.mkdir(parents=True, exist_ok=True)
    (scorecards / "v1.yaml").write_text(text)
    (scorecards / "reference_full.yaml").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=list(PAIRS), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-parquet", type=Path, default=Path("var/bias_in_bios.parquet"),
                        help="local cache of the downloaded parquet, so a rerun need not refetch")
    args = parser.parse_args()
    out_dir = args.out or Path("fixtures/bios_pairs") / args.pair

    if args.cache_parquet.exists():
        print(f"reading cached {args.cache_parquet}")
        df = pd.read_parquet(args.cache_parquet)
    else:
        df = download_dataframe()
        args.cache_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.cache_parquet)

    spec = PAIRS[args.pair]
    less_female_id, less_female = spec["less_female"]
    more_female_id, more_female = spec["more_female"]
    labels = {less_female_id: less_female, more_female_id: more_female}

    sampled = sample_labels(df, labels)
    items, twins = build_items(sampled, labels)

    write_items(items, out_dir / "items.jsonl")
    write_scorecards(out_dir, less_female, more_female)

    counts = {name: len(sampled[label_id]) for label_id, name in labels.items()}
    print(f"wrote {len(items)} items to {out_dir / 'items.jsonl'}: {counts}")
    print(f"{len(twins)} counterfactual twins (one per item -- every item is held out)")
    print(f"wrote scorecards to {out_dir / 'scorecards'}")


if __name__ == "__main__":
    main()
