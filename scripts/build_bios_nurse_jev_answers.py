#!/usr/bin/env python
"""Assemble ``fixtures/bios_nurse/answers.jsonl.gz``, spending on only what's new.

    python scripts/build_bios_nurse_jev_answers.py --price-only
    python scripts/build_bios_nurse_jev_answers.py

Copied from ``scripts/build_bios_attorney_jev_answers.py`` (owned by the paralegal/attorney
agent, so not imported directly) with the fixtures pointed at the nurse/physician pair.

The held-out nurse/physician bios already have Jev answers
(``fixtures/bios_pairs/nurse_physician/answers.jsonl.gz``). This study's fixtures
(``fixtures/bios_nurse/items.jsonl``, built by ``build_bios_nurse_fixtures.py``) reuse those
same held-out ids and texts, add a new labeling pool, and regenerate the held-out twins under
the amended swap rule -- only 20 of 2,000 twins actually changed text. So instead of asking Jev
again for everything, this script:

1. copies the existing held-out + unchanged-twin rows by id from the nurse/physician fixture;
2. asks Jev only for the pool items and the twins whose text changed, via
   ``scripts.build_bios_jev_answers`` (priced first, same money rule; logs to
   ``studies/bios_nurse_spend.md``).

Writes ``fixtures/bios_nurse/answers.jsonl.gz`` in the same extract shape as every other bios
answers file.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bios_jev_answers import run as run_jev  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

OLD_FIXTURES = Path("fixtures/bios_pairs/nurse_physician")
NEW_FIXTURES = Path("fixtures/bios_nurse")
SPEND_LOG = Path("studies/bios_nurse_spend.md")


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def log_spend(step: str, items: int, priced: int, sent: int, notes: str) -> None:
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not SPEND_LOG.exists():
        SPEND_LOG.write_text("| when | step | items | requests priced | requests sent | notes |\n"
                             "|---|---|---:|---:|---:|---|\n")
    row = f"| {date.today()} | {step} | {items} | {priced} | {sent} | {notes} |\n"
    with SPEND_LOG.open("a", encoding="utf-8") as handle:
        handle.write(row)
    print(f"[spend] {step}: priced {priced}, sent {sent} -- {notes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    load_dotenv()

    new_items = {row["id"]: row for row in load_jsonl(NEW_FIXTURES / "items.jsonl")}
    old_items = {row["id"]: row for row in load_jsonl(OLD_FIXTURES / "items.jsonl")}
    old_answers = {row["id"]: row for row in load_jsonl(OLD_FIXTURES / "answers.jsonl.gz")}

    reusable = []
    to_ask = []
    for item_id, row in new_items.items():
        split = row["metadata"]["split"]
        if split == "pool":
            to_ask.append(item_id)
            continue
        old_row = old_answers.get(item_id)
        old_item = old_items.get(item_id)
        if old_row is None or old_item is None or old_item["text"] != row["text"]:
            to_ask.append(item_id)
            continue
        reusable.append(item_id)

    n_pool = sum(1 for i in to_ask if new_items[i]["metadata"]["split"] == "pool")
    n_twins = sum(1 for i in to_ask if new_items[i]["metadata"]["split"] != "pool")
    print(f"{len(new_items)} items total; {len(reusable)} reusable from the existing "
          f"nurse/physician answers; {len(to_ask)} need a fresh Jev answer "
          f"({n_pool} pool, {n_twins} changed twins)")
    log_spend("pool + changed held-out twins: priced", len(to_ask), len(to_ask), 0,
              f"priced before sending; {n_pool} pool (physician 2000, nurse 2000) + {n_twins} "
              f"twins whose text changed under the amended swap rule")

    subset_path = Path("var/bios_nurse_to_ask.jsonl")
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8") as handle:
        for item_id in to_ask:
            handle.write(json.dumps(new_items[item_id], ensure_ascii=False) + "\n")

    fresh_out = Path("var/bios_nurse_fresh_answers.jsonl.gz")
    fresh_partial = Path("var/bios_nurse-jev-answers.partial.jsonl")
    asyncio.run(run_jev(NEW_FIXTURES, fresh_out, fresh_partial, None, args.price_only,
                        args.concurrency, subset_path))
    if args.price_only:
        return

    out = NEW_FIXTURES / "answers.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as dst:
        for item_id in reusable:
            dst.write(json.dumps(old_answers[item_id], ensure_ascii=False) + "\n")
        for row in load_jsonl(fresh_out):
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    log_spend("pool + changed held-out twins: sent", len(to_ask), len(to_ask), len(to_ask),
              "see var/bios_nurse-jev-answers.partial.jsonl for per-item usage")


if __name__ == "__main__":
    main()
