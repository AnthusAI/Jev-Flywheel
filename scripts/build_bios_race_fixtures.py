#!/usr/bin/env python
"""Build the race-name counterfactual fixtures for the held-out bios.

    python scripts/build_bios_race_fixtures.py

Reads ``fixtures/bios/items.jsonl``, takes the 2,000 ``split == "test"`` items (sorted by id
for a reproducible draw), and for each calls ``jev_flywheel.names.name_versions`` with one
``random.Random(0)`` instance created once and advanced in item order. A bio with no subject
pronoun (``he``/``she``) cannot carry a name and is excluded, and counted.

Writes ``fixtures/bios/race_versions.jsonl``: one row per surviving version (three per bio --
``white_a``, ``white_b``, ``black``), each with ``id`` = ``<item id>-<version>``, ``text``, and
``metadata`` carrying ``version``, ``name``, ``source_id``, ``occupation``, ``gender``,
``reference_label``. See ``studies/PREREGISTERED.md``, "does the engine read race from a name?"
for the method and the pre-registered predictions.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from jev_flywheel.items import load_items
from jev_flywheel.names import name_versions

VERSIONS = ("white_a", "white_b", "black")


def build(items_path: Path, out_path: Path) -> None:
    items = [i for i in load_items(items_path) if i.split == "test"]
    items.sort(key=lambda i: i.id)
    print(f"{len(items)} test items")

    rng = random.Random(0)
    rows = []
    excluded = 0
    for item in items:
        versions = name_versions(item.text, item.metadata["gender"], rng)
        if versions is None:
            excluded += 1
            continue
        for version in VERSIONS:
            text = getattr(versions, version)
            name = versions.names[version]
            rows.append({
                "id": f"{item.id}-{version}",
                "text": text,
                "metadata": {
                    "version": version,
                    "name": name,
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

    print(f"{excluded} of {len(items)} items excluded (no subject pronoun)")
    print(f"{n_bios} bios kept, {len(rows)} versions written "
          f"(expect {n_bios} x 3 = {n_bios * 3})")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=Path("fixtures/bios/items.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("fixtures/bios/race_versions.jsonl"))
    args = parser.parse_args()
    build(args.items, args.out)


if __name__ == "__main__":
    main()
