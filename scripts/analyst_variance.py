#!/usr/bin/env python
"""How much does one steering round vary, and does the analyst model matter?

Each run starts from a fresh workspace, labels 140 items with the simulated labeler (a
different seed per run, so the labeled set differs), runs one real steering round, and scores
every scorecard version on the same fixed sample of held-out items. The analyst is
non-deterministic, so a single run is one draw; this reports several.

    python scripts/analyst_variance.py --env .env \\
        --models us.moonshotai.kimi-k3 moonshotai.kimi-k2.5 --seeds 1 2 3 4

A model can carry its region as ``model@region`` (Qwen3 Coder 480B is only in some regions):
``qwen.qwen3-coder-480b-a35b-v1:0@us-west-2``.

Costs real money on both sides (Bedrock for the analyst, Jev for the top-ups): roughly 740
Jev requests and one LLM call per run. Results are appended to studies/analyst_variance.jsonl
so an interrupted study resumes rather than repeats.
"""
import argparse
import asyncio
import json
import random
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv

from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.jev import JevSession
from jev_flywheel.report import complete_items, history
from jev_flywheel.simulate import label_with_reference
from jev_flywheel.steer import ScriptedApprover, run_steering
from jev_flywheel.workspace import Workspace


def one_run(spec: str, seed: int, labels: int, sample: int) -> dict:
    model, _, region = spec.partition("@")     # "model@region"; region optional
    work = Path(tempfile.mkdtemp(prefix="flywheel-study-"))
    workspace = Workspace.init(work / "var", PACKAGED_FIXTURES)
    score = workspace.scorecard().scores[0].name
    label_with_reference(workspace, score, labels, seed=seed)
    result = {"model": spec, "seed": seed, "labels": workspace.n_labeled(score)}
    outcome = run_steering(workspace, score, model=model, region=region or None, allow_spend=True,
                           hitl_handler=ScriptedApprover(default=True))
    result["decision"] = outcome.decision
    result["root_cause"] = outcome.detail.get("root_cause")
    if not outcome.promoted:
        result["detail"] = {k: v for k, v in outcome.detail.items() if k != "root_cause"}
        return result

    card = workspace.scorecard()
    result["elements"] = [e.key for e in card.score(score).elements]
    # A different held-out sample per run: reusing one fixed 600 across every run and model
    # would make the cross-model comparison selection on a single sample.
    picked = random.Random(hash((spec, seed)) % (2**31)).sample(workspace.split("test"), sample)
    report = asyncio.run(workspace.cache.fill(JevSession(), picked, card.questions()))
    result["topup_failures"] = report.failures
    items = complete_items(workspace)
    result["n_scored"] = len(items)
    result["versions"] = {
        f"v{p.version}": {"kind": p.kind, "accuracy": round(p.scoreboard.summary.accuracy, 4),
                          "ece": round(p.scoreboard.summary.ece, 4),
                          "brier": round(p.scoreboard.summary.brier, 4)}
        for p in history(workspace, score, item_ids=items)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["us.moonshotai.kimi-k3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--labels", type=int, default=140)
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("studies/analyst_variance.jsonl"))
    args = parser.parse_args()
    load_dotenv(args.env)          # python-dotenv: the value is never echoed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["model"], row["seed"]))
    for model in args.models:
        for seed in args.seeds:
            if (model, seed) in done:
                print(f"skip {model} seed {seed} (already recorded)")
                continue
            try:
                row = one_run(model, seed, args.labels, args.sample)
            except Exception as error:  # noqa: BLE001 - one bad run must not sink the study
                row = {"model": model, "seed": seed, "decision": "error",
                       "error": f"{type(error).__name__}: {error}"[:300]}
                traceback.print_exc()
            with args.out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            versions = row.get("versions", {})
            last = versions[max(versions)] if versions else None
            first = versions.get("v1")
            print(f"{model:28s} seed {seed}: {row['decision']:<20s}"
                  + (f" acc {first['accuracy']:.3f} -> {last['accuracy']:.3f}, "
                     f"ECE {first['ece']:.3f} -> {last['ece']:.3f}" if last else ""))


if __name__ == "__main__":
    main()
