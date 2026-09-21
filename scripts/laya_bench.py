#!/usr/bin/env python
"""Measure the engine facts the Laya chapter states, instead of asserting them.

    python scripts/laya_bench.py

Four things, each with a claim in ``studies/PREREGISTERED.md`` that this either supports or
contradicts:

1. **Marginal cost of a question.** Latency for 1..12 questions about the same item. Jev's
   cost of an extra question is a few input tokens; Laya re-encodes the state once per row.
2. **Determinism.** The same request twice in one process, and once in each of two fresh
   processes. Answers are compared exactly.
3. **Sibling-independence.** Whether a question's answer depends on which other questions rode
   in the request. The per-question answer cache is sound only if it does not. Rows are padded
   to a common length inside a batch, so exact equality is not guaranteed and the size of the
   difference is the finding.
4. **Window headroom.** The longest item in the corpus against the tokens Laya has left.

Latency is only meaningful on a quiet machine, so the load average is recorded beside it and
the report says when it was not quiet. Results go to ``studies/laya_bench.json``.
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.items import load_items
from jev_flywheel.laya import DEFAULT_CHECKPOINT, check_budget, to_laya_question
from jev_flywheel.scorecard import Scorecard

QUIET_LOAD = 2.0            # 1-minute load average above this is not a quiet machine


def load_questions():
    fixtures = Path(PACKAGED_FIXTURES)
    card = Scorecard.from_yaml((fixtures / "scorecards" / "reference_full.yaml").read_text())
    return ({n: to_laya_question(q) for n, q in card.questions().items()},
            load_items(fixtures / "items.jsonl"))


def probabilities(answer: dict) -> list:
    """Every probability an answer carries, in a fixed order, for comparing two of them."""
    if "probabilities" in answer:
        return [answer["probabilities"][k] for k in sorted(answer["probabilities"])]
    return [answer["noul"]]


def marginal_cost(agent, questions, items, reps: int):
    noul = {f"n{i}": {"type": "noul", "instructions": f"Does the text express feeling number {i}?"}
            for i in range(12)}
    texts = [i.text for i in items[:10]]
    out = []
    for k in range(1, 13):
        qs = dict(list(noul.items())[:k])
        for text in texts[:3]:
            agent.system_one(text, qs)                               # warm up
        samples = []
        for _ in range(reps):
            for text in texts:
                t0 = time.perf_counter()
                agent.system_one(text, qs)
                samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        out.append({"questions": k, "median_ms": round(statistics.median(samples), 2),
                    "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2)})
    return out


def determinism_in_process(agent, questions, items):
    a = agent.system_one(items[0].text, questions)["answers"]
    b = agent.system_one(items[0].text, questions)["answers"]
    return a == b


def fingerprint_answers(checkpoint: str) -> str:
    """Run in a fresh process: hash the answers to a fixed set of items."""
    import laya_mlx
    questions, items = load_questions()
    agent = laya_mlx.load(checkpoint)
    digest = hashlib.sha256()
    for item in items[:50]:
        digest.update(json.dumps(agent.system_one(item.text, questions)["answers"],
                                 sort_keys=True).encode())
    return digest.hexdigest()


def determinism_across_processes(checkpoint: str) -> bool:
    cmd = [sys.executable, __file__, "--fingerprint", "--checkpoint", checkpoint]
    hashes = {subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
              for _ in range(2)}
    return len(hashes) == 1


def sibling_independence(agent, questions, items, n_items: int, seed: int = 0):
    """Ask each question alone and among its siblings; report how far the answers move."""
    rng = random.Random(seed)
    sample = rng.sample(items, n_items)
    names = list(questions)
    deltas, exact, total = [], 0, 0
    order_deltas = []
    for item in sample:
        together = agent.system_one(item.text, questions)["answers"]
        shuffled = names[:]
        rng.shuffle(shuffled)
        reordered = agent.system_one(item.text, {n: questions[n] for n in shuffled})["answers"]
        for name in names:
            alone = agent.system_one(item.text, {name: questions[name]})["answers"][name]
            d = max(abs(x - y) for x, y in zip(probabilities(alone), probabilities(together[name])))
            o = max(abs(x - y) for x, y in
                    zip(probabilities(together[name]), probabilities(reordered[name])))
            deltas.append(d)
            order_deltas.append(o)
            exact += d == 0.0
            total += 1
    deltas.sort()
    return {
        "items": n_items, "comparisons": total, "exactly_equal": exact,
        "share_exactly_equal": round(exact / total, 4),
        "max_abs_diff": max(deltas), "p99_abs_diff": deltas[int(0.99 * (len(deltas) - 1))],
        "mean_abs_diff": round(statistics.mean(deltas), 6),
        "max_abs_diff_when_only_the_order_changes": max(order_deltas),
    }


def window_headroom(agent, questions, items):
    tok = agent.tok
    longest = max(items, key=lambda i: len(i.text))
    counts = [len(tok(i.text, add_special_tokens=False)["input_ids"]) for i in items]
    check_budget(agent, longest.text, questions)          # raises if any question would truncate
    from laya_mlx.common import build_prefix
    room = min(agent.cfg.get("max_len", 512) - len(build_prefix(
        tok, agent._to_internal(q), agent.cfg.get("head_max_len", 192))[0]) - 1
        for q in questions.values())
    return {"items": len(items), "longest_item_chars": len(longest.text),
            "max_state_tokens": max(counts), "mean_state_tokens": round(statistics.mean(counts), 1),
            "tightest_room_across_the_eight_questions": room}


def main(out: Path, checkpoint: str, reps: int, sibling_items: int) -> None:
    import laya_mlx
    questions, items = load_questions()
    load = os.getloadavg()[0]
    agent = laya_mlx.load(checkpoint)
    report = {
        "checkpoint": checkpoint, "machine_load_1min": round(load, 2),
        "quiet": load <= QUIET_LOAD,
        "runtime": "laya-mlx (independent MLX port, not an official release)",
        "marginal_cost": marginal_cost(agent, questions, items, reps),
        "deterministic_in_process": determinism_in_process(agent, questions, items),
        "deterministic_across_processes": determinism_across_processes(checkpoint),
        "sibling_independence": sibling_independence(agent, questions, items, sibling_items),
        "window_headroom": window_headroom(agent, questions, items),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["quiet"]:
        print(f"WARNING: load average {load:.1f} is above {QUIET_LOAD}; latencies are not a benchmark")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("studies/laya_bench.json"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--sibling-items", type=int, default=150)
    parser.add_argument("--fingerprint", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fingerprint:
        print(fingerprint_answers(args.checkpoint))
    else:
        main(args.out, args.checkpoint, args.reps, args.sibling_items)
