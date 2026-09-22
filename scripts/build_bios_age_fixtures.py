#!/usr/bin/env python
"""Build the age-insertion counterfactual fixtures for the held-out bios.

    python scripts/build_bios_age_fixtures.py

Reads ``fixtures/bios/items.jsonl``, takes the 2,000 ``split == "test"`` items (sorted by id
for a reproducible run), and keeps the ones ``jev_flywheel.age.eligible`` accepts: a subject
pronoun to insert at, no year before 2000, and no stated duration of ten or more years. For
each eligible bio, four ages (34, 35, 61, 62) are inserted with
``jev_flywheel.age.insert_age``.

Writes ``fixtures/bios/age_versions.jsonl``: one row per version, four per bio, each with
``id`` = ``<item id>-age<N>``, ``text``, and ``metadata`` carrying ``age``, ``case``,
``source_id``, ``occupation``, ``gender``, ``reference_label``. See
``studies/PREREGISTERED.md``, "does the engine read age?" for the method and the
pre-registered predictions -- 1,231 bios are expected to be eligible (640 surgeon,
591 physician).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_flywheel.age import eligible, insert_age
from jev_flywheel.items import load_items

AGES = (34, 35, 61, 62)


def build(items_path: Path, out_path: Path) -> None:
    items = [i for i in load_items(items_path) if i.split == "test"]
    items.sort(key=lambda i: i.id)
    print(f"{len(items)} test items")

    rows = []
    excluded = 0
    by_occupation: dict = {}
    for item in items:
        if not eligible(item.text):
            excluded += 1
            continue
        by_occupation[item.metadata["occupation"]] = (
            by_occupation.get(item.metadata["occupation"], 0) + 1)
        for age in AGES:
            insertion = insert_age(item.text, age)
            assert insertion is not None, f"{item.id} passed eligible() but insert_age failed"
            rows.append({
                "id": f"{item.id}-age{age}",
                "text": insertion.text,
                "metadata": {
                    "age": age,
                    "case": insertion.case,
                    "source_id": item.id,
                    "occupation": item.metadata["occupation"],
                    "gender": item.metadata["gender"],
                    "reference_label": item.metadata["reference_label"],
                },
            })

    n_bios = len(items) - excluded
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{excluded} of {len(items)} items excluded (ineligible: no subject pronoun, a year "
          f"before 2000, or a duration of 10+ years)")
    print(f"{n_bios} bios kept, by occupation: {by_occupation}")
    print(f"{len(rows)} versions written (expect {n_bios} x 4 = {n_bios * 4})")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=Path("fixtures/bios/items.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("fixtures/bios/age_versions.jsonl"))
    args = parser.parse_args()
    build(args.items, args.out)


if __name__ == "__main__":
    main()
