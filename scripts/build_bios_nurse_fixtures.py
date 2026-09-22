#!/usr/bin/env python
"""Build the fixtures for "the learning loop on nurse vs physician"
(``studies/PREREGISTERED.md``, final section): the nurse/physician pair, with a labeling pool,
so the flywheel loop (J1/J2/L1/L2) has something to learn from.

    python scripts/build_bios_nurse_fixtures.py

Two pieces, both written to ``fixtures/bios_nurse/``:

1. **The existing 1,000 + 1,000 held-out bios**, copied by id from
   ``fixtures/bios_pairs/nurse_physician/items.jsonl`` (same ids, same redacted texts -- nothing
   about the held-out sample changes), but with every counterfactual twin *regenerated* under
   the amended ``swap_gender`` (the "women's/women's health" protection and the Miss/Sir/Madam
   additions, already in ``jev_flywheel/counterfactual.py``). A twin's text only changes if the
   amended rule touches something the old rule didn't; both counts are printed and recorded.
2. **A new labeling pool**: 2,000 bios per label (physician, nurse), sampled uniformly at
   random from the train split, seed 1, split ``"pool"``, disjoint from every id used in *any*
   earlier bios fixture in this repo -- not just this pair's own held-out sample. The
   paralegal/attorney loop found that ids collide across studies (they are all drawn from the
   same underlying parquet, indexed by row number), so disjointness here is checked against the
   union of ``fixtures/bios/items.jsonl``, every ``fixtures/bios_pairs/*/items.jsonl``, and
   ``fixtures/bios_attorney/items.jsonl`` -- every fixture that existed before this study ran.
   Redacted and swapped (amended rule) the same way.

``scorecards/v1.yaml`` and ``reference_full.yaml`` are copied unchanged from the nurse/physician
pair fixtures (same one question, same positive class "physician"). ``first_names.txt`` is not
written here either -- redaction reuses ``fixtures/bios/first_names.txt`` directly, exactly as
``build_bios_pairs_fixtures.py`` and ``build_bios_attorney_fixtures.py`` do.

The raw parquet is cached at ``var/bias_in_bios.parquet`` (shared with the other bios fixture
builders; downloaded if missing).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd

from jev_flywheel.counterfactual import redact_names_batch, swap_gender

PARQUET_URL = ("https://huggingface.co/api/datasets/LabHC/bias_in_bios/"
               "parquet/default/train/0.parquet")

SOURCE = Path("fixtures/bios_pairs/nurse_physician")
OUT = Path("fixtures/bios_nurse")
LABELS = {19: "physician", 13: "nurse"}   # same ids as build_bios_pairs_fixtures.PAIRS
POSITIVE = "physician"
N_PER_LABEL_POOL = 2000
POOL_SEED = 1

# Every fixture file that could already hold ids drawn from the same parquet, checked before
# this study's own pool draw touches any of them.
PRIOR_FIXTURE_GLOBS = [
    "fixtures/bios/items.jsonl",
    "fixtures/bios_pairs/*/items.jsonl",
    "fixtures/bios_attorney/items.jsonl",
]

_INDEX_RE = re.compile(r"(\d+)(?:-swapped)?$")


def download_dataframe() -> pd.DataFrame:
    print(f"downloading {PARQUET_URL} ...")
    data = urllib.request.urlopen(PARQUET_URL, timeout=120).read()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"  {len(df):,} rows, columns {df.columns.tolist()}")
    return df


def load_dataframe(cache: Path) -> pd.DataFrame:
    if cache.exists():
        print(f"reading cached {cache}")
        return pd.read_parquet(cache)
    df = download_dataframe()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def regenerate_held_out(source_items: List[dict]) -> tuple[List[dict], int, int]:
    """Same ids, same base texts; twins regenerated under the amended swap rule.

    Returns (items, n_twins_changed, n_twins_total).
    """
    by_id = {row["id"]: row for row in source_items}
    out_items = []
    changed = 0
    total = 0
    for row in source_items:
        if row["metadata"].get("split") == "counterfactual":
            continue  # rebuilt below, from the (unchanged) original text
        out_items.append(row)
        twin_id = f"{row['id']}-swapped"
        old_twin = by_id.get(twin_id)
        if old_twin is None:
            continue  # a pool item in the source fixtures has no twin (there is no pool there)
        swap = swap_gender(row["text"])
        total += 1
        new_twin_text = swap.text
        if new_twin_text != old_twin["text"]:
            changed += 1
        new_twin = {
            "id": twin_id, "text": new_twin_text,
            "metadata": dict(old_twin["metadata"], swapped=swap.swapped,
                             her_resolved=swap.her_resolved),
        }
        out_items.append(new_twin)
    return out_items, changed, total


def _index_of(item_id: str) -> int | None:
    match = _INDEX_RE.search(item_id)
    return int(match.group(1)) if match else None


def prior_fixture_indices() -> set:
    """Every parquet row index used by any fixture that existed before this study, across all
    the globs in ``PRIOR_FIXTURE_GLOBS``. Ids not matching ``bios...<digits>`` are ignored (none
    are expected in these files)."""
    indices = set()
    files = []
    for pattern in PRIOR_FIXTURE_GLOBS:
        files.extend(sorted(glob.glob(pattern)))
    seen_files = []
    for path_str in files:
        path = Path(path_str)
        if path.resolve() == (OUT / "items.jsonl").resolve():
            continue  # this study's own (not-yet-written or stale) output, never an input
        seen_files.append(path)
        for row in load_jsonl(path):
            idx = _index_of(row["id"])
            if idx is not None:
                indices.add(idx)
    print(f"prior-fixture disjointness check: {len(indices):,} distinct parquet indices across "
          f"{len(seen_files)} files: {[str(p) for p in seen_files]}")
    return indices


def sample_pool(df: pd.DataFrame, exclude_indices: set) -> pd.DataFrame:
    frames = []
    for label in LABELS:
        pool = df[df["profession"] == label]
        pool = pool[~pool.index.isin(exclude_indices)]
        n = min(N_PER_LABEL_POOL, len(pool))
        if n < N_PER_LABEL_POOL:
            print(f"  label {label} ({LABELS[label]}) has only {n} rows available after "
                  f"excluding prior-fixture ids; using all {n}, below the requested "
                  f"{N_PER_LABEL_POOL}")
        sampled = pool.sample(n=n, random_state=POOL_SEED)
        sampled = sampled.copy()
        sampled["split"] = "pool"
        frames.append(sampled)
    return pd.concat(frames)


def build_pool_items(df: pd.DataFrame) -> List[dict]:
    raw_texts = [str(row["hard_text"]) for _, row in df.iterrows()]
    print(f"redacting names from {len(raw_texts)} pool bios (spaCy en_core_web_sm) ...")
    redactions = redact_names_batch(raw_texts)
    items = []
    for (idx, row), redaction in zip(df.iterrows(), redactions):
        occupation = LABELS[int(row["profession"])]
        item_id = f"bios-pool-{idx:06d}"
        metadata = {
            "split": "pool",
            "reference_label": occupation,
            "occupation": occupation,
            "gender": "female" if int(row["gender"]) == 1 else "male",
            "redacted": redaction.redacted,
        }
        items.append({"id": item_id, "text": redaction.text, "metadata": metadata})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-parquet", type=Path, default=Path("var/bias_in_bios.parquet"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    source_items = load_jsonl(SOURCE / "items.jsonl")
    held_out, twins_changed, twins_total = regenerate_held_out(source_items)
    n_held_out_bios = sum(1 for r in held_out if r["metadata"]["split"] == "test")
    print(f"held-out bios: {n_held_out_bios}; twins regenerated: {twins_total}; "
          f"changed under the amended rule: {twins_changed} "
          f"({twins_changed / twins_total:.2%})" if twins_total else "")

    df = load_dataframe(args.cache_parquet)
    exclude = prior_fixture_indices()
    pool_df = sample_pool(df, exclude)
    overlap = set(pool_df.index) & exclude
    assert not overlap, f"pool draw overlaps a prior fixture's ids: {sorted(overlap)[:5]}"
    pool_items = build_pool_items(pool_df)

    all_items = held_out + pool_items
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_items:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    scorecards_dir = args.out / "scorecards"
    scorecards_dir.mkdir(parents=True, exist_ok=True)
    for name in ("v1.yaml", "reference_full.yaml"):
        shutil.copyfile(SOURCE / "scorecards" / name, scorecards_dir / name)

    by_split: Dict[str, int] = {}
    for row in all_items:
        by_split[row["metadata"]["split"]] = by_split.get(row["metadata"]["split"], 0) + 1
    counts_by_label = {}
    for row in pool_items:
        occ = row["metadata"]["occupation"]
        counts_by_label[occ] = counts_by_label.get(occ, 0) + 1

    print(f"wrote {len(all_items)} items to {args.out / 'items.jsonl'}: {by_split}")
    print(f"pool composition: {counts_by_label}")
    print(f"wrote scorecards to {scorecards_dir}")
    print(json.dumps({
        "held_out_bios": n_held_out_bios,
        "held_out_twins_regenerated": twins_total,
        "held_out_twins_changed": twins_changed,
        "pool_size": len(pool_items),
        "pool_by_label": counts_by_label,
    }, indent=2))


if __name__ == "__main__":
    main()
