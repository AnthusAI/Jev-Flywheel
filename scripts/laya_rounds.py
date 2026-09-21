#!/usr/bin/env python
"""Several steering rounds, with a local Laya answering every question and a hosted analyst steering.

    aws login                                  # the analyst is a Bedrock model
    python scripts/laya_rounds.py --seeds 1 2 3

The recorded run stops after one steering round with 140 labels; that is how the recording was
scripted, not what the method needs. This is the independent arm the paired study left open: no
Jev answers are used at all. Laya answers the holistic question and every element the analyst
proposes (locally, free), and the analyst reads the disagreements and proposes the next
question. After each labeling budget one steering round runs, so accuracy can be followed as
the question set grows and the labels grow with it.

    labels:  140 -> round 1 -> 300 -> round 2 -> 500 -> round 3 -> 800 -> round 4

Every version is scored on the README's 600 held-out items and on all 3,521 (Laya is free, so
there is no reason not to). The proposals are recorded in full so they can be judged by reading:
did round 2 find something round 1 did not? Cost: one to two Bedrock calls per round, and no Jev
calls at all. Results append to ``studies/laya_rounds.jsonl``; a finished seed is skipped.
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
from jev_flywheel.laya import LayaClient
from jev_flywheel.proposal import ProposalError, parse_proposal
from jev_flywheel.report import history
from jev_flywheel.simulate import label_with_reference
from jev_flywheel.steer import ScriptedApprover, run_steering
from jev_flywheel.workspace import Workspace

PAPER_SAMPLE = 600


def proposals(outcome):
    out = []
    for reply in filter(None, (outcome.analyst_reply, outcome.discovery_reply)):
        try:
            for a in parse_proposal(reply).add:
                out.append({"key": a.key, "question_type": a.question_type,
                            "instructions": a.instructions,
                            "criteria": getattr(a, "criteria", None)})
        except ProposalError:
            pass
    return out


def score_versions(ws, score, factory):
    """Every version, on the README's 600 held-out items and on all of them. Laya is free.

    Each version is scored with *its own* questions. A round may reword an element, and the
    answers to the old wording are not the answers to the new one; scoring an earlier version
    with only the final card's answers silently penalises it.
    """
    test = ws.split("test")
    session = JevSession(client_factory=factory)
    for version in range(1, ws.version + 1):
        asyncio.run(ws.cache.fill(session, test, ws.scorecard(version).questions()))
    paper = {i.id for i in random.Random(0).sample(test, PAPER_SAMPLE)}
    versions = {}
    for name, ids in (("paper600", paper), ("full", {i.id for i in test})):
        for p in history(ws, score, item_ids=ids):
            versions.setdefault(f"v{p.version}", {"kind": p.kind, "labels": p.n_feedback})[name] = {
                "accuracy": round(p.scoreboard.summary.accuracy, 4),
                "ece": round(p.scoreboard.summary.ece, 4),
                "brier": round(p.scoreboard.summary.brier, 4),
                "coverage": round(p.scoreboard.coverage, 4)}
    return versions


def one_run(client, model, seed, budgets, region):
    work = Path(tempfile.mkdtemp(prefix="laya-rounds-"))
    ws = Workspace.init(work / "var", PACKAGED_FIXTURES, answers="answers-laya.jsonl.gz",
                        engine="laya")
    score = ws.scorecard().scores[0].name
    factory = lambda: client  # noqa: E731
    rounds, labeled = [], 0
    for number, budget in enumerate(budgets, 1):
        label_with_reference(ws, score, budget - labeled, seed=seed * 100 + number)
        labeled = ws.n_labeled(score)
        outcome = run_steering(ws, score, model=model, region=region or None, allow_spend=True,
                               client_factory=factory, hitl_handler=ScriptedApprover(default=True),
                               max_auto_requests=100000)
        rounds.append({"round": number, "labels": labeled, "decision": outcome.decision,
                       "root_cause": outcome.detail.get("root_cause"),
                       "why_not": outcome.detail.get("reason") or outcome.detail.get("problem"),
                       "proposed": proposals(outcome),
                       "elements": [e.key for e in ws.scorecard().score(score).elements]})
        print(f"  seed {seed} round {number}: {labeled} labels, {outcome.decision}, "
              f"elements now {rounds[-1]['elements']}", flush=True)

    versions = score_versions(ws, score, factory)
    return {"engine": "laya", "model": model, "seed": seed, "budgets": budgets,
            "rounds": rounds, "versions": versions}


def rescore(out, pairs):
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    client = LayaClient()
    client.warm()
    for pair in pairs:
        seed, _, directory = pair.partition("=")
        ws = Workspace(Path(directory) / "var")
        score = ws.scorecard().scores[0].name
        for row in rows:
            if row["seed"] == int(seed) and row.get("versions"):
                row["versions"] = score_versions(ws, score, lambda: client)
                print(f"rescored seed {seed}: {len(row['versions'])} versions")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="us.moonshotai.kimi-k3")
    parser.add_argument("--region", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--budgets", type=int, nargs="+", default=[140, 300, 500, 800])
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("studies/laya_rounds.jsonl"))
    parser.add_argument("--rescore", nargs="+", default=[], metavar="SEED=WORKSPACE_DIR",
                        help="recompute the scores of recorded seeds from their saved workspaces "
                             "(no analyst calls), rewriting those rows in --out")
    args = parser.parse_args()
    load_dotenv(args.env)          # python-dotenv: the value is never echoed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.rescore:
        rescore(args.out, args.rescore)
        return
    done = set()
    if args.out.exists():
        rows = [json.loads(line) for line in args.out.read_text().splitlines()]
        done = {r["seed"] for r in rows if r.get("versions")}
    client = LayaClient()
    client.warm()
    for seed in args.seeds:
        if seed in done:
            print(f"skip seed {seed} (already recorded)")
            continue
        try:
            row = one_run(client, args.model, seed, args.budgets, args.region)
        except Exception as error:  # noqa: BLE001 - one bad run must not sink the study
            traceback.print_exc()
            row = {"engine": "laya", "model": args.model, "seed": seed, "decision": "error",
                   "error": f"{type(error).__name__}: {error}"[:300]}
        with args.out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        for name, v in sorted(row.get("versions", {}).items(), key=lambda kv: int(kv[0][1:])):
            print(f"  seed {seed} {name:3} {v['kind']:6} {v['labels']:>4} labels  "
                  f"600: {v['paper600']['accuracy']:.3f}  full: {v['full']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
