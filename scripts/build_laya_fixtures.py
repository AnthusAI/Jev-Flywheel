#!/usr/bin/env python
"""Answer the whole corpus with a local Laya, in the same shape as the Jev fixture.

    python scripts/build_laya_fixtures.py

Asks the seven-element reference scorecard's eight questions about every item, one request per
item (one forward pass per question inside it), and writes ``fixtures/answers-laya.jsonl.gz``:
one ``{"id", "model", "usage", "answers"}`` row per item, the format ``import_answers_jsonl``
reads and ``fixtures/answers.jsonl.gz`` already uses. Runs locally, costs nothing, and resumes
if interrupted (a partial file is kept in the scratch path and completed rows are skipped).

Unlike Jev's extract, this one is reproducible: Laya is deterministic, so running this again
produces the same answers. The rows also carry ``latency_ms`` for the cost comparison.

Requires ``pip install 'jev-flywheel[laya]'`` and Apple silicon, and downloads the ~843 MB
checkpoint from Hugging Face on first use.
"""
import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.items import load_items
from jev_flywheel.jev import JevSession
from jev_flywheel.laya import LayaClient
from jev_flywheel.scorecard import Scorecard


async def main(out: Path, partial: Path, limit: int | None) -> None:
    fixtures = Path(PACKAGED_FIXTURES)
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
          f"{len(questions)} questions each")

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
            if n % 500 == 0:
                rate = n / (time.perf_counter() - started)
                print(f"  {n}/{len(todo)}  {rate:.1f} items/s", flush=True)

    with partial.open(encoding="utf-8") as src, gzip.open(out, "wt", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("fixtures/answers-laya.jsonl.gz"))
    parser.add_argument("--partial", type=Path,
                        default=Path("laya-answers.partial.jsonl"),
                        help="working file, so an interrupted run resumes")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (a check)")
    args = parser.parse_args()
    asyncio.run(main(args.out, args.partial, args.limit))
