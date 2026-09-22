#!/usr/bin/env python
"""Answer every Bias-in-Bios item (and its counterfactual twin) with Jev -- the J0 arm's input.

    python scripts/build_bios_jev_answers.py --price-only     # count requests, spend nothing
    python scripts/build_bios_jev_answers.py                  # send them

Asks the v1 scorecard's one question ("Is this person a surgeon or a physician?") of every item
in ``fixtures/bios/items.jsonl`` (pool, test, and the test items' counterfactual twins) via
``TYPESAFE_API_KEY``, and writes ``fixtures/bios/answers.jsonl.gz`` in the same
``{"id", "model", "usage", "answers"}`` extract shape ``scripts/build_bios_laya_answers.py`` uses,
so ``scripts/run_bios_arms.py --arm J0`` can score it exactly like L0.

Priced first, per ``studies/PREREGISTERED.md``'s money rule: with no flag, this only reports how
many requests the run needs and exits without spending anything. Neither ``typesafe-sdk`` 0.7.0
nor this repo's ``flywheel topup`` exposes a dollar rate anywhere (checked before writing this
script) -- the "price" this reports, like ``flywheel topup`` without ``--yes``, is a request and
input-token count, which is what ``studies/bios_gender_spend.md`` records against the study's
budget in lieu of a $/request rate.

Resumable: a partial file under ``var/`` is kept and completed rows are skipped on a rerun, same
as the Laya script.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from jev_flywheel.items import load_items
from jev_flywheel.jev import JevSession
from jev_flywheel.scorecard import Scorecard


async def run(fixtures: Path, out: Path, partial: Path, limit: int | None, price_only: bool,
              concurrency: int) -> None:
    card = Scorecard.from_yaml((fixtures / "scorecards" / "reference_full.yaml").read_text())
    questions = card.questions()
    items = load_items(fixtures / "items.jsonl")
    if limit:
        items = items[:limit]

    done = set()
    if partial.exists():
        done = {json.loads(line)["id"] for line in partial.read_text().splitlines() if line}
    todo = [item for item in items if item.id not in done]
    print(f"{len(items)} items, {len(done)} already answered, {len(todo)} to go; "
          f"{len(questions)} question(s) each")
    if not todo:
        if partial.exists() and not out.exists():
            _finish(partial, out)
        return
    if price_only:
        print(f"would send {len(todo)} Jev requests (1 per item; see "
              f"studies/bios_gender_spend.md for the running total). Not spending anything.")
        return

    session = JevSession()  # default client_factory: builds the real typesafe-sdk client
    semaphore = asyncio.Semaphore(concurrency)
    partial.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n_done = 0
    write_lock = asyncio.Lock()
    handle = partial.open("a", encoding="utf-8")

    async def one(item):
        nonlocal n_done
        async with semaphore:
            t0 = time.perf_counter()
            try:
                result = await session.ask(item.text, questions)
            except Exception as error:  # noqa: BLE001 - one bad item must not sink the run
                print(f"  FAILED {item.id}: {type(error).__name__}: {error}")
                return
            row = {"id": item.id, "model": result.model, "usage": result.usage,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                   "answers": result.answers}
            async with write_lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                n_done += 1
                if n_done % 500 == 0:
                    rate = n_done / (time.perf_counter() - started)
                    print(f"  {n_done}/{len(todo)}  {rate:.1f} items/s", flush=True)

    await asyncio.gather(*(one(item) for item in todo))
    handle.close()
    print(f"sent {session.requests_sent} requests, {session.input_tokens:,} input tokens, "
          f"{session.output_tokens:,} output tokens")
    _finish(partial, out)


def _finish(partial: Path, out: Path) -> None:
    with partial.open(encoding="utf-8") as src, gzip.open(out, "wt", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures/bios"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--partial", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--price-only", action="store_true",
                        help="report how many requests would be sent and exit; spend nothing")
    args = parser.parse_args()
    out = args.out or args.fixtures / "answers.jsonl.gz"
    partial = args.partial or Path("var") / f"{args.fixtures.name}-jev-answers.partial.jsonl"
    load_dotenv()
    asyncio.run(run(args.fixtures, out, partial, args.limit, args.price_only, args.concurrency))


if __name__ == "__main__":
    main()
