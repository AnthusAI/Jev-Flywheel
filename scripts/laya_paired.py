#!/usr/bin/env python
"""The same recorded run, replayed against Jev's answers and against Laya's.

    python scripts/laya_paired.py

The human's labels are a property of the *items*, not of whoever predicted them, so one
recording (140 labels, the refit points, and the analyst's proposal of a `topic_domain`
element) can be replayed against two engines. Everything is held fixed except who answers the
questions, so the difference between the two lineages is the engine and nothing else.

* **jev**: the recording as committed. Offline; its answers are the recorded ones.
* **laya**: the same labels and steps against a local Laya. The proposed element is *asked of
  Laya* (free, on this machine) rather than restored from Jev's answers.

Each version is scored twice: on the 600 held-out items the README's Jev numbers use
(``random.Random(0).sample(test, 600)``, so the comparison is item-for-item), and, for Laya
only, on all of the held-out split, which is affordable only because Laya is free. Rows go to
``studies/laya_paired.jsonl`` with per-tier accuracy and answer coverage, so the comparison can
be mined later rather than only summarised. Nothing here is fitted to the held-out items.
"""
import argparse
import asyncio
import json
import random
import tempfile
from pathlib import Path

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.jev import JevSession
from jev_flywheel.laya import LayaClient
from jev_flywheel.recording import replay
from jev_flywheel.report import complete_items, history
from jev_flywheel.workspace import Workspace

RECORDING = Path(PACKAGED_FIXTURES) / "recordings" / "simulated-labeler"
PAPER_SAMPLE = 600     # the README's held-out sample, drawn with seed 0


def rows_for(engine: str, workspace: Workspace, score: str, item_ids, sample: str):
    for point in history(workspace, score, item_ids=item_ids):
        s = point.scoreboard
        yield {
            "engine": engine, "sample": sample, "version": point.version, "kind": point.kind,
            "n_feedback": point.n_feedback, "n_items": len(item_ids),
            "accuracy": round(s.summary.accuracy, 4), "ece": round(s.summary.ece, 4),
            "brier": round(s.summary.brier, 4), "coverage": round(s.coverage, 4),
            "by_tier": {t: round(v, 4) for t, v in s.by_tier.items()},
        }


def main(out: Path) -> None:
    fixtures = Path(PACKAGED_FIXTURES)
    rows = []

    # --- Jev: the recording as committed -------------------------------------------------
    jev = replay(RECORDING, Path(tempfile.mkdtemp(prefix="paired-jev-")) / "var", fixtures)
    score = jev.scorecard().scores[0].name
    paper = complete_items(jev, "test")
    assert len(paper) == PAPER_SAMPLE, f"expected the README's {PAPER_SAMPLE} items, got {len(paper)}"
    rows += rows_for("jev", jev, score, paper, "paper-600")

    # --- Laya: the same labels and steps, answered locally --------------------------------
    client = LayaClient()
    client.warm()
    laya = replay(RECORDING, Path(tempfile.mkdtemp(prefix="paired-laya-")) / "var", fixtures,
                  answers="answers-laya.jsonl.gz", engine="laya",
                  client_factory=lambda: client, on_step=lambda m: None)
    versions = [p.version for p in history(laya, score)]
    print(f"laya lineage: {len(versions)} versions ({versions})")

    # Free, so ask about every held-out item, not only the 600.
    test = laya.split("test")
    questions = laya.scorecard().questions()
    report = asyncio.run(laya.cache.fill(JevSession(client_factory=lambda: client), test, questions))
    print(f"laya top-up over {len(test)} test items: {report.requested} asked, "
          f"{report.failures} failed")

    same_600 = {i.id for i in random.Random(0).sample(jev.split("test"), PAPER_SAMPLE)}
    assert same_600 == paper, "the two engines must be scored on the identical 600 items"
    rows += rows_for("laya", laya, score, same_600, "paper-600")
    rows += rows_for("laya", laya, score, {i.id for i in test}, "full-test")

    # What a local engine spent, in its own units.
    stats = client.stats
    rows.append({"engine": "laya", "kind": "cost", "calls": stats.calls, "rows": stats.rows,
                 "input_tokens": stats.input_tokens, "seconds": round(stats.seconds, 1)})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:       # a study is one run: rewrite, not append
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"\n{'engine':5} {'sample':10} {'ver':>3} {'kind':6} {'labels':>6} "
          f"{'acc':>6} {'ECE':>6} {'Brier':>6} {'cover':>6}")
    for r in rows:
        if "accuracy" in r:
            print(f"{r['engine']:5} {r['sample']:10} v{r['version']:<2} {r['kind']:6} "
                  f"{r['n_feedback']:>6} {r['accuracy']:>6.3f} {r['ece']:>6.3f} {r['brier']:>6.3f} "
                  f"{r['coverage']:>6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("studies/laya_paired.jsonl"))
    main(parser.parse_args().out)
