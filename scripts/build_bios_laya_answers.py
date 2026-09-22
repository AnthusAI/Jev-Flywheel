#!/usr/bin/env python
"""Answer every Bias-in-Bios item (and its counterfactual twin) with local Laya.

    python scripts/build_bios_laya_answers.py
    python scripts/build_bios_laya_answers.py --fixtures fixtures/bios_nurse
    python scripts/build_bios_laya_answers.py --items fixtures/bios/race_versions.jsonl \
        --out fixtures/bios/answers-race-laya.jsonl.gz

Asks the v1 scorecard's one question ("Is this person a surgeon or a physician?") of every item
in ``fixtures/bios/items.jsonl`` (pool, test, and the test items' counterfactual twins), and
writes ``fixtures/bios/answers-laya.jsonl.gz`` in the same ``{"id", "model", "usage", "answers"}``
extract shape ``scripts/build_laya_fixtures.py`` uses for the sentiment corpus. Free, local,
deterministic, and resumable (a partial file under ``var/`` is kept and completed rows skipped).

Jev's answers (``answers.jsonl.gz``) are not produced here: this study's pre-registration needs
``TYPESAFE_API_KEY``, which was not available when this was run (see the pre-registration's
deviations note).
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

from jev_flywheel.items import load_items
from jev_flywheel.jev import JevSession
from jev_flywheel.laya import LayaClient
from jev_flywheel.scorecard import Scorecard


async def run(fixtures: Path, out: Path, partial: Path, limit: int | None,
              items_path: Path | None = None) -> None:
    card = Scorecard.from_yaml((fixtures / "scorecards" / "reference_full.yaml").read_text())
    questions = card.questions()
    items = load_items(items_path or fixtures / "items.jsonl")
    if limit:
        items = items[:limit]

    done = set()
    if partial.exists():
        done = {json.loads(line)["id"] for line in partial.read_text().splitlines() if line}
    todo = [item for item in items if item.id not in done]
    print(f"{len(items)} items, {len(done)} already answered, {len(todo)} to go; "
          f"{len(questions)} question(s) each")

    client = LayaClient()
    client.warm()
    session = JevSession(client_factory=lambda: client)
    partial.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with partial.open("a", encoding="utf-8") as handle:
        for n, item in enumerate(todo, 1):
            t0 = time.perf_counter()
            result = await session.ask(item.text, questions)
            handle.write(json.dumps({
                "id": item.id, "model": result.model, "usage": result.usage,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "answers": result.answers}, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 1000 == 0:
                rate = n / (time.perf_counter() - started)
                print(f"  {n}/{len(todo)}  {rate:.1f} items/s", flush=True)

    with partial.open(encoding="utf-8") as src, gzip.open(out, "wt", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main(fixtures: Path, out: Path, partial: Path, limit: int | None,
         items_path: Path | None = None) -> None:
    asyncio.run(run(fixtures, out, partial, limit, items_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/bios"))
    parser.add_argument("--items", type=Path, default=None,
                        help="items file to answer (default: <fixtures>/items.jsonl)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--partial", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out = args.out or args.fixtures / "answers-laya.jsonl.gz"
    partial = args.partial or Path("var") / f"{args.fixtures.name}-laya-answers.partial.jsonl"
    main(args.fixtures, out, partial, args.limit, args.items)
