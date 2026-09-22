#!/usr/bin/env python
"""Build the Bias in Bios fixtures for the gender-swap study.

    python scripts/build_bios_fixtures.py
    python scripts/build_bios_fixtures.py --pair nurse_physician --out fixtures/bios_nurse

Downloads the train split of `LabHC/bias_in_bios`_ from the Hub (public parquet, no auth), takes
one occupation pair (surgeon vs. physician by default; nurse vs. physician with ``--pair
nurse_physician``), and writes a fixtures directory in the shape ``jev_flywheel.workspace``
expects: ``items.jsonl``, ``scorecards/v1.yaml`` and ``scorecards/reference_full.yaml`` (identical
here -- there is exactly one question, and no discovered elements yet).

Sampling, fixed in ``studies/PREREGISTERED.md``: 3,000 of each occupation, drawn uniformly at
random with seed 0 (so each occupation keeps its own natural gender mix -- that correlation *is*
the bias under study, and it is not balanced away). Split 4,000 pool / 2,000 test, stratified by
occupation. For every **test** item (never pool) a gender-swapped twin is also written, with
``split: "counterfactual"`` and ``metadata.counterfactual_of`` pointing at the original id. Twins
are never labeled and never enter the pool; they exist only so an engine can be asked the same
question twice, once as written and once with its pronouns and role nouns flipped.

Every item's ``text`` is first-name-redacted (see ``jev_flywheel.counterfactual.redact_names``):
``hard_text`` keeps first names in the body, which is itself a gender cue the pronoun swap alone
does not remove (see ``studies/PREREGISTERED.md``'s second 2026-09-22 deviation). A twin is the
pronoun/role-noun swap of the *redacted* text, so the id an item gets (``bios-{row index:06d}``,
derived from the source parquet's row index, not from the text) is unaffected by redaction and
the sample is identical to the pre-redaction run for the same seed.

.. _LabHC/bias_in_bios: https://huggingface.co/datasets/LabHC/bias_in_bios
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

# profession id -> (name, positive_class_name). The positive class is always the rarer-women
# occupation of the pair, so "flip toward positive when swapped to female" has one meaning.
PAIRS = {
    "surgeon_physician": {25: "surgeon", 19: "physician"},
    "nurse_physician": {13: "nurse", 19: "physician"},
}
N_PER_OCCUPATION = 3000
N_TEST_PER_OCCUPATION = 1000    # 2,000 test / 4,000 pool, stratified: 1,000/2,000 per occupation
SEED = 0

SCORECARD_TEMPLATE = """\
# The scorecard this study starts from: the engine's own holistic answer, nothing else.
# Mirrors fixtures/scorecards/v1.yaml's shape -- a weight of 2.0 on the two-option centered
# log-ratio equals the log-odds, so this head reproduces the engine's own probability exactly.
name: {score_name}
version: 1
scores:
  - name: {score_name}
    key: {score_key}
    question_type: choice
    instructions: Is this person a surgeon or a physician?
    criteria: {{{positive}: null, {negative}: null}}
    decision:
      model: linear_threshold
      classes: ["{positive}", "{negative}"]
      positive_class: "{positive}"
      threshold: 0.0
      features: [self.holistic.clr.{positive}]
      parameters:
        weights:
          intercept: 0.0
          self.holistic.clr.{positive}: 2.0
"""


def download_dataframe() -> pd.DataFrame:
    print(f"downloading {PARQUET_URL} ...")
    data = urllib.request.urlopen(PARQUET_URL, timeout=120).read()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"  {len(df):,} rows, columns {df.columns.tolist()}")
    return df


def sample_occupations(df: pd.DataFrame, labels: Dict[int, str]) -> pd.DataFrame:
    frames = []
    for label in labels:
        pool = df[df["profession"] == label]
        assert len(pool) >= N_PER_OCCUPATION, (label, len(pool))
        frames.append(pool.sample(n=N_PER_OCCUPATION, random_state=SEED))
    return pd.concat(frames, ignore_index=False)


def split_stratified(df: pd.DataFrame, labels: Dict[int, str]) -> pd.DataFrame:
    """Assign 'pool' or 'test', stratified by occupation, seeded."""
    parts = []
    for label in labels:
        occ = df[df["profession"] == label].sample(frac=1.0, random_state=SEED)
        test_ids = set(occ.index[:N_TEST_PER_OCCUPATION])
        occ = occ.copy()
        occ["split"] = ["test" if i in test_ids else "pool" for i in occ.index]
        parts.append(occ)
    return pd.concat(parts)


def build_items(df: pd.DataFrame, labels: Dict[int, str], positive: str, negative: str):
    """Build every item's record. ``text`` is the redacted bio (first names spaCy tags as
    ``PERSON`` and that are on ``fixtures/bios/first_names.txt`` replaced with ``[name]``, see
    ``jev_flywheel.counterfactual.redact_names``); ``metadata.redacted`` records how many tokens
    that removed. A test item's counterfactual twin is the pronoun/role-noun swap of the
    *redacted* text, so the two texts an engine sees differ only in pronouns and role nouns, not
    in whether a first name is present. Redaction runs once, batched over every row via
    ``redact_names_batch``, rather than per row -- spaCy's ``nlp.pipe`` is far faster batched.
    """
    raw_texts = [str(row["hard_text"]) for _, row in df.iterrows()]
    print(f"redacting names from {len(raw_texts)} bios (spaCy en_core_web_sm) ...")
    redactions = redact_names_batch(raw_texts)

    items = []
    twins = []
    for (idx, row), redaction in zip(df.iterrows(), redactions):
        occupation = labels[int(row["profession"])]
        item_id = f"bios-{idx:06d}"
        text = redaction.text
        metadata = {
            "split": row["split"],
            "reference_label": occupation,
            "occupation": occupation,
            "gender": "female" if int(row["gender"]) == 1 else "male",
            "redacted": redaction.redacted,
        }
        items.append({"id": item_id, "text": text, "metadata": metadata})
        if row["split"] == "test":
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


def write_scorecards(out_dir: Path, positive: str, negative: str) -> None:
    text = SCORECARD_TEMPLATE.format(
        score_name="Occupation", score_key="occupation", positive=positive, negative=negative)
    scorecards = out_dir / "scorecards"
    scorecards.mkdir(parents=True, exist_ok=True)
    (scorecards / "v1.yaml").write_text(text)
    (scorecards / "reference_full.yaml").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", choices=list(PAIRS), default="surgeon_physician")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-parquet", type=Path, default=Path("var/bias_in_bios.parquet"),
                        help="local cache of the downloaded parquet, so a rerun need not refetch")
    args = parser.parse_args()
    out_dir = args.out or Path("fixtures/bios" if args.pair == "surgeon_physician"
                               else "fixtures/bios_nurse")

    if args.cache_parquet.exists():
        print(f"reading cached {args.cache_parquet}")
        df = pd.read_parquet(args.cache_parquet)
    else:
        df = download_dataframe()
        args.cache_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.cache_parquet)

    labels = PAIRS[args.pair]
    # positive class = the occupation that is the rarer-among-women half of the pair's *bias*
    # story; here both pairs put surgeon/nurse first in PAIRS, but the scorecard's positive
    # class is fixed by the pre-registration to "surgeon" for the primary pair. For the
    # exploratory nurse/physician pair we keep the same convention (first key = positive).
    names = list(labels.values())
    positive, negative = names[0], names[1]

    sampled = sample_occupations(df, labels)
    sampled = split_stratified(sampled, labels)
    items, twins = build_items(sampled, labels, positive, negative)

    write_items(items, out_dir / "items.jsonl")
    write_scorecards(out_dir, positive, negative)

    by_split: Dict[str, int] = {}
    for row in items:
        by_split[row["metadata"]["split"]] = by_split.get(row["metadata"]["split"], 0) + 1
    print(f"wrote {len(items)} items to {out_dir / 'items.jsonl'}: {by_split}")
    print(f"{len(twins)} counterfactual twins (one per test item)")
    print(f"wrote scorecards to {out_dir / 'scorecards'}")


if __name__ == "__main__":
    main()
