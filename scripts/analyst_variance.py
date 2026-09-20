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


# The corpus has a deliberately planted latent factor, documented by the project that built
# it: sports contexts skew positive, workplace contexts skew negative ("a learnable domain
# pattern ... a demonstration of task-specific alignment"). An element NAMES it only if it
# asks about subject matter on both poles; asking "is this bureaucratic?" is a proxy for the
# workplace half and does not count. The flag is a screen, not a verdict: `instructions` is
# recorded in full so the judgement can be made by reading.
SPORTS = ("sport", "athlet", "recreation", "game", "team", "fitness", "exercise", "leisure")
WORK = ("workplace", "office", "work-related", "business", "corporate", "professional",
        "administrative", "job")
SUBJECT = ("about", "subject", "topic", "domain", "context", "setting")


def names_the_plant(elements) -> bool:
    """Symmetric: either pole counts, provided it is asked as a question about subject matter.

    The earlier version required a sports word, so an element naming only the workplace pole
    could never score -- an asymmetry that manufactured a "0/12". The flag stays a screen:
    `instructions` is recorded in full so the judgement can be made by reading.
    """
    for e in elements:
        text = f"{e.get('key','')} {e.get('instructions','')}".lower()
        poles = any(s in text for s in SPORTS) + any(w in text for w in WORK)
        if poles == 2:
            return True
        if poles == 1 and any(s in text for s in SUBJECT):
            return True
    return False


def one_run(spec: str, seed: int, labels: int, sample: int, max_mismatches: int = 25,
            arm: str = "d0") -> dict:
    model, _, region = spec.partition("@")     # "model@region"; region optional
    work = Path(tempfile.mkdtemp(prefix="flywheel-study-"))
    workspace = Workspace.init(work / "var", PACKAGED_FIXTURES)
    score = workspace.scorecard().scores[0].name
    label_with_reference(workspace, score, labels, seed=seed)
    result = {"model": spec, "seed": seed, "arm": arm, "labels": workspace.n_labeled(score)}
    outcome = run_steering(workspace, score, model=model, region=region or None, allow_spend=True,
                           hitl_handler=ScriptedApprover(default=True),
                           max_mismatches=max_mismatches,
                           discovery=arm in ("d1", "d2"), taxonomy=arm == "d2")
    result["decision"] = outcome.decision
    result["root_cause"] = outcome.detail.get("root_cause")
    # Use the real parser: models wrap JSON in prose and code fences, and a bare json.loads
    # silently recorded those runs as having proposed nothing.
    from jev_flywheel.proposal import ProposalError, parse_proposal
    proposed = []
    for reply in filter(None, (outcome.analyst_reply, outcome.discovery_reply)):
        try:
            for a in parse_proposal(reply).add:
                proposed.append({"key": a.key, "question_type": a.question_type,
                                 "instructions": a.instructions})
        except ProposalError:
            pass
    result["proposed"] = [{"key": e.get("key"), "instructions": e.get("instructions")}
                          for e in proposed]
    result["found_the_plant"] = names_the_plant(proposed)
    if not outcome.promoted:
        result["detail"] = {k: v for k, v in outcome.detail.items() if k != "root_cause"}
        return result

    card = workspace.scorecard()
    result["elements"] = [e.key for e in card.score(score).elements]
    # A different held-out sample per run: reusing one fixed 600 across every run and model
    # would make the cross-model comparison selection on a single sample.
    # One held-out sample per SEED, shared by every model and arm, so the comparison is
    # paired. (Re-drawing per run made v1 range 0.747-0.795 and cross-run numbers noise.)
    picked = random.Random(seed).sample(workspace.split("test"), sample)
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
    parser.add_argument("--arm", default="d0", choices=["d0", "d1", "d2"],
                        help="d0: error analysis only (the current loop). d1: + a frame-free "
                             "discovery call on a blind Group A/B sample. d2: d1 plus a "
                             "taxonomy of convention kinds in the prompt.")
    parser.add_argument("--max-mismatches", type=int, default=25,
                        help="How many disagreements the analyst is shown. A pattern spread "
                             "across a corpus is hard to see in a short list.")
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("studies/analyst_variance.jsonl"))
    args = parser.parse_args()
    load_dotenv(args.env)          # python-dotenv: the value is never echoed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["model"], row["seed"], row.get("arm", "d0")))
    for model in args.models:
        for seed in args.seeds:
            if (model, seed, args.arm) in done:
                print(f"skip {model} seed {seed} (already recorded)")
                continue
            try:
                row = one_run(model, seed, args.labels, args.sample, args.max_mismatches,
                              args.arm)
            except Exception as error:  # noqa: BLE001 - one bad run must not sink the study
                row = {"model": model, "seed": seed, "decision": "error",
                       "error": f"{type(error).__name__}: {error}"[:300]}
                traceback.print_exc()
            with args.out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            versions = row.get("versions", {})
            last = versions[max(versions)] if versions else None
            first = versions.get("v1")
            print(f"{model:28s} seed {seed}: {'PLANT' if row.get('found_the_plant') else '     '} "
                  f"{row['decision']:<20s}"
                  + (f" acc {first['accuracy']:.3f} -> {last['accuracy']:.3f}, "
                     f"ECE {first['ece']:.3f} -> {last['ece']:.3f}" if last else ""))


if __name__ == "__main__":
    main()
