#!/usr/bin/env python
"""Build the full-name race counterfactual fixtures for the second attempt.

    python scripts/build_bios_race2_fixtures.py

Reads ``fixtures/bios/items.jsonl`` (the 2,000 ``split == "test"`` items, sorted by id) and
``fixtures/bios/name_pools.json`` (built by ``scripts/build_name_pools.py``), and for every bio
draws 4 (first, last) name pairs per group -- white, black, hispanic, asian -- gender-matched to
the bio, from one ``random.Random(0)`` advanced in item-id order across every bio (eligible or
not, matching ``scripts/build_bios_race_fixtures.py``'s precedent: the draw happens before
eligibility is checked, so a later bio's names never shift because an earlier one turned out to
have no insertion point). A bio with no insertion point at all (see
``jev_flywheel.fullname.analyze_full_name``) is excluded, and counted.

Writes:

* ``fixtures/bios/race2_versions.jsonl`` -- one row per surviving version (16 per bio: 4 names
  x 4 groups), id ``<item id>-<group>-<k>``, with ``metadata`` carrying ``group``, ``k``,
  ``first``, ``last``, ``source_id``, ``occupation``, ``gender``, ``reference_label``.
* ``fixtures/bios/race2_jev_subsample.txt`` -- 500 eligible source ids, drawn uniformly by a
  second, independent ``random.Random(0)`` (selecting *which bios* Jev answers is a separate
  concern from *which names* every bio gets, so it gets its own seeded stream).

See ``studies/PREREGISTERED.md``, "race from a full name, second attempt", points 3-4.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from jev_flywheel.fullname import analyze_full_name, render_full_name
from jev_flywheel.items import load_items

GROUPS = ("white", "black", "hispanic", "asian")
NAMES_PER_GROUP = 4
SUBSAMPLE_SIZE = 500


def _load_pools(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))["groups"]


def build(items_path: Path, pools_path: Path, out_versions: Path, out_subsample: Path) -> None:
    items = [i for i in load_items(items_path) if i.split == "test"]
    items.sort(key=lambda i: i.id)
    print(f"{len(items)} test items")

    pools = _load_pools(pools_path)

    rng = random.Random(0)
    rows: List[Dict] = []
    eligible_ids: List[str] = []
    excluded = 0

    for item in items:
        gender = item.metadata["gender"]
        plan = analyze_full_name(item.text)

        drawn = []  # (group, k, first, last), always drawn, even if the bio is excluded below
        for group in GROUPS:
            first_group = "white" if pools[group]["uses_white_first_pool"] else group
            first_pool = pools[first_group]["first"][gender]
            last_pool = pools[group]["last"]
            for k in range(1, NAMES_PER_GROUP + 1):
                first = rng.choice(first_pool)
                last = rng.choice(last_pool)
                drawn.append((group, k, first, last))

        if not plan.has_insertion_point:
            excluded += 1
            continue

        eligible_ids.append(item.id)
        for group, k, first, last in drawn:
            result = render_full_name(plan, first, last)
            rows.append({
                "id": f"{item.id}-{group}-{k}",
                "text": result.text,
                "metadata": {
                    "group": group,
                    "k": k,
                    "first": first,
                    "last": last,
                    "source_id": item.id,
                    "occupation": item.metadata["occupation"],
                    "gender": gender,
                    "reference_label": item.metadata["reference_label"],
                },
            })

    n_bios = len(items) - excluded
    out_versions.parent.mkdir(parents=True, exist_ok=True)
    with out_versions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{excluded} of {len(items)} items excluded (no insertion point)")
    print(f"{n_bios} bios kept, {len(rows)} versions written "
          f"(expect {n_bios} x {len(GROUPS) * NAMES_PER_GROUP} = "
          f"{n_bios * len(GROUPS) * NAMES_PER_GROUP})")
    print(f"wrote {out_versions}")

    sub_rng = random.Random(0)
    subsample = sub_rng.sample(sorted(eligible_ids), min(SUBSAMPLE_SIZE, len(eligible_ids)))
    subsample.sort()
    out_subsample.parent.mkdir(parents=True, exist_ok=True)
    out_subsample.write_text("\n".join(subsample) + "\n", encoding="utf-8")
    print(f"{len(subsample)} of {len(eligible_ids)} eligible bios sampled for Jev")
    print(f"wrote {out_subsample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=Path("fixtures/bios/items.jsonl"))
    parser.add_argument("--pools", type=Path, default=Path("fixtures/bios/name_pools.json"))
    parser.add_argument("--out-versions", type=Path,
                        default=Path("fixtures/bios/race2_versions.jsonl"))
    parser.add_argument("--out-subsample", type=Path,
                        default=Path("fixtures/bios/race2_jev_subsample.txt"))
    args = parser.parse_args()
    build(args.items, args.pools, args.out_versions, args.out_subsample)


if __name__ == "__main__":
    main()
