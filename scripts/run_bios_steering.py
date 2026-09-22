#!/usr/bin/env python
"""L1/L2/J1/J2: the flywheel as it stands, and with the invariance gate, on the bios corpus.

    python scripts/run_bios_steering.py --arm L1 --seeds 1 2 3
    python scripts/run_bios_steering.py --arm J2 --seeds 1 2 3

``studies/PREREGISTERED.md``'s "does the engine read gender, and can the layer refuse to?"
section. Follows ``scripts/laya_rounds.py``'s pattern (label, one steering round, score on
held-out) adapted to ``fixtures/bios``'s held-out "test" split and its gender-swapped
"counterfactual" twins, and to a single 140-label round rather than four budgets.

Each seed: 140 labels from the corpus's own occupation label (the simulated labeler), one
steering round (the invariance gate on for L2/J2, off for L1/J1), scored with
``scripts/bios_gender.py`` on the 2,000 held-out bios and their twins. Every proposed element's
wording is appended to ``studies/bios_gender_proposals.jsonl`` regardless of outcome, and every
arm's metrics row to ``studies/bios_gender.jsonl``, matching ``scripts/run_bios_arms.py``'s J0/L0
rows.

The session (140 labels, every refit, the steering round, and the answers a promoted element
needed on the labeled and held-out items and their twins) is recorded to
``fixtures/bios/recordings/<arm>-seed<N>/`` via ``jev_flywheel.recording``, so a reader with no
keys (a Laya recording: no GPU either -- see that module's engine-aware baseline) can replay the
whole arm offline.

Money (``studies/PREREGISTERED.md``'s rule): Laya (L1/L2) is free. Jev (J1/J2) spends on three
things, each logged to ``studies/bios_gender_spend.md`` before it is sent: up to 140 requests to
top up the 140 labeled items with a newly proposed element's answer (every arm that proposes
something), up to another 140 for J2's gate (the labeled items' swapped twins), and -- only if
the element is promoted -- up to 4,000 to serve it on the 2,000 held-out bios and their 2,000
twins so the arm can be scored. An arm that proposes nothing, or whose proposal is rejected,
spends only the first of those and never the 4,000.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, score_arm, write_rows  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from jev_flywheel.jev import JevSession, _default_client  # noqa: E402
from jev_flywheel.proposal import ProposalError, parse_proposal  # noqa: E402
from jev_flywheel.recording import record  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.simulate import label_with_reference  # noqa: E402
from jev_flywheel.steer import ScriptedApprover, run_steering  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

FIXTURES = Path("fixtures/bios")
RECORDINGS = FIXTURES / "recordings"
N_LABELS = 140
GATE_RATE = 0.02
SCORE_NAME = "Occupation"

ARM_CONFIG = {
    "L1": {"engine": "laya", "gate": False},
    "L2": {"engine": "laya", "gate": True},
    "J1": {"engine": "jev", "gate": False},
    "J2": {"engine": "jev", "gate": True},
}
ANSWER_FILES = {"jev": "answers.jsonl.gz", "laya": "answers-laya.jsonl.gz"}
SPEND_LOG = Path("studies/bios_gender_spend.md")
PROPOSALS_OUT = Path("studies/bios_gender_proposals.jsonl")
METRICS_OUT = Path("studies/bios_gender.jsonl")


class CountingClient:
    """Wraps a real client, counting requests and tokens -- so the spend log can report what a
    steering round actually spent, not just what a plan estimated (the plan does not know
    whether a top-up will hit the invariance gate's own twin-answering path, which asks the
    workspace's cache directly rather than going through this same client instance...

    ...except that it *does*: ``FlywheelHost`` builds every ``JevSession`` it needs from the
    same ``client_factory`` this script hands ``run_steering``, so one counter here sees every
    request a round sends, evaluate top-up, gate top-up and all.
    """

    def __init__(self, inner):
        self._inner = inner
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def system_one(self, *, state, questions):
        response = await self._inner.system_one(state=state, questions=questions)
        self.requests += 1
        usage = response.usage
        usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        return response

    def check_questions(self, questions, text):
        inner_check = getattr(self._inner, "check_questions", None)
        if inner_check is not None:
            inner_check(questions, text)


def log_spend(when: str, step: str, items: int, priced: int, sent: int, notes: str) -> None:
    row = f"| {when} | {step} | {items} | {priced} | {sent} | {notes} |\n"
    with SPEND_LOG.open("a", encoding="utf-8") as handle:
        handle.write(row)
    print(f"[spend] {step}: priced {priced}, sent {sent} -- {notes}")


_LAYA_CLIENT = None


def laya_client():
    """One warm Laya client for the whole script run: reloading the checkpoint per seed would
    cost a minute of model loading six times over for nothing -- Laya is deterministic, so
    sharing one instance across seeds changes nothing about the numbers."""
    global _LAYA_CLIENT
    if _LAYA_CLIENT is None:
        from jev_flywheel.laya import LayaClient

        _LAYA_CLIENT = LayaClient()
        _LAYA_CLIENT.warm()
    return _LAYA_CLIENT


def client_factory_for(engine: str, counter: Optional[CountingClient] = None):
    if engine == "laya":
        return laya_client
    # jev: a single real client, wrapped once so every JevSession this script or the host
    # builds (they all take this same factory) counts against one running total.
    return lambda: counter


def wire_flip_rate(flip_rates: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if not flip_rates:
        return None
    for wire, rate in flip_rates.items():
        if wire == key or wire.endswith(f".{key}"):
            return rate
    return None


def proposal_rows(arm: str, engine: str, seed: int, outcome, gate_active: bool
                  ) -> List[Dict[str, Any]]:
    """One row per proposed element, whatever happened to it.

    ``passed_gate`` is computed directly from the measured flip rate against the fixed 0.02
    threshold (``jev_flywheel.invariance.DEFAULT_MAX_FLIP_RATE``), not by pattern-matching the
    prose in ``outcome.detail["reasons"]``: that text is ``host._evaluation_text``'s whole
    multi-line summary (out-of-fold numbers *and* the rejection reasons run together), and
    splitting it on ";" to isolate one element's reason is fragile -- it mis-parsed a
    two-element proposal's gate failures on the first bios_gender run this script made,
    reporting the gate as passed when the recorded flip rate was 24x the 2% limit. ``passed_fit``
    (the *ordinary* out-of-fold test, independent of the gate) is derived the same direct way,
    by checking for the two other reason templates ``jev_flywheel.fit.compare`` ever writes.
    """
    if not outcome.analyst_reply:
        return []
    try:
        proposal = parse_proposal(outcome.analyst_reply)
    except ProposalError:
        return [{"arm": arm, "engine": engine, "seed": seed, "key": None,
                 "wording": outcome.analyst_reply, "passed_fit": None,
                 "flip_rate_on_labeled": None, "passed_gate": None, "promoted": False,
                 "decision": outcome.decision, "note": "unparseable analyst reply"}]
    reasons_text = str(outcome.detail.get("reasons") or "")
    non_gate_reason_present = ("Brier improved by" in reasons_text
                               or "accuracy fell by" in reasons_text)
    rows = []
    for added in proposal.add:
        flip = wire_flip_rate(outcome.invariance_flip_rates, added.key)
        passed_gate = (flip <= GATE_RATE) if (gate_active and flip is not None) else None
        if outcome.decision == "promoted":
            passed_fit = True
        elif outcome.decision == "rejected_by_metrics":
            passed_fit = not non_gate_reason_present
        else:
            passed_fit = None
        rows.append({
            "arm": arm, "engine": engine, "seed": seed, "key": added.key,
            "wording": added.instructions, "question_type": added.question_type,
            "passed_fit": passed_fit, "flip_rate_on_labeled": flip,
            "passed_gate": passed_gate, "promoted": outcome.decision == "promoted",
            "decision": outcome.decision, "root_cause": proposal.root_cause})
    if not proposal.add:
        rows.append({
            "arm": arm, "engine": engine, "seed": seed, "key": None, "wording": None,
            "passed_fit": None, "flip_rate_on_labeled": None, "passed_gate": None,
            "promoted": False, "decision": outcome.decision, "root_cause": proposal.root_cause,
            "note": "no element proposed" if outcome.decision != "invalid_proposal"
                    else "invalid proposal"})
    return rows


def score_held_out(ws: Workspace, arm: str, engine: str, seed: int, factory) -> tuple:
    """Score the fitted (or unchanged) scorecard on the 2,000 held-out bios and their twins.

    Prices the top-up before sending it (the money rule): missing answers are only ever for a
    newly promoted element, since the holistic question is fully cached from
    ``fixtures/bios/answers{,-laya}.jsonl.gz``.
    """
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    test_items = ws.split("test")
    twin_items = ws.split("counterfactual")
    all_items = test_items + twin_items
    plan = ws.cache.plan([i.id for i in all_items], questions)
    sent = 0
    if plan.requests:
        # studies/bios_gender_spend.md is the Jev spend log (the money rule's $60/25,000-request
        # cap is a Jev-only concern); Laya is free and local, so a Laya top-up is never logged
        # there, or a reader would (wrongly) read it as Jev spend.
        if engine == "jev":
            log_spend(str(date.today()), f"{arm} seed {seed}: serve held-out + twins",
                      len(all_items), plan.requests, 0, "priced before sending")
        session = JevSession(client_factory=factory)
        report = asyncio.run(ws.cache.fill(session, all_items, questions))
        sent = report.requested
        if engine == "jev":
            log_spend(str(date.today()), f"{arm} seed {seed}: serve held-out + twins (sent)",
                      len(all_items), plan.requests, sent,
                      f"{report.failures} failures" if report.failures else "0 failures")

    def verdict(item) -> Verdict:
        answers = ws.cache.partial_answers_for(item.id, questions)
        result = predict(score, answers)
        confidence = result.confidence if result.confidence is not None else 0.5
        p_surgeon = confidence if result.value == "surgeon" else 1.0 - confidence
        return Verdict(item.id, result.value, p_surgeon, item.reference_label,
                       item.metadata.get("gender"))

    verdicts = [verdict(i) for i in test_items]
    twins = {i.metadata["counterfactual_of"]: verdict(i) for i in twin_items}
    metrics = score_arm(arm=arm, engine=engine, verdicts=verdicts, twins=twins, seed=seed,
                        version=card.version, n_labels=ws.n_labeled(SCORE_NAME), redacted=True)
    return metrics, sent


def one_run(arm: str, seed: int, workspace_root: Path) -> Dict[str, Any]:
    config = ARM_CONFIG[arm]
    engine, gate = config["engine"], config["gate"]
    counter = CountingClient(_default_client()) if engine == "jev" else None
    factory = client_factory_for(engine, counter)

    ws = Workspace.init(workspace_root / "var", FIXTURES, answers=ANSWER_FILES[engine],
                        engine=engine)
    label_with_reference(ws, SCORE_NAME, N_LABELS, seed=seed)
    assert ws.n_labeled(SCORE_NAME) == N_LABELS, ws.n_labeled(SCORE_NAME)

    if engine == "jev":
        log_spend(str(date.today()), f"{arm} seed {seed}: steering round (eval top-up{'+ gate' if gate else ''})",
                  N_LABELS, N_LABELS * (2 if gate else 1), 0,
                  "priced as an upper bound before the round (the exact top-up depends on "
                  "whether an element is proposed)")

    outcome = run_steering(
        ws, SCORE_NAME, allow_spend=True, client_factory=factory,
        hitl_handler=ScriptedApprover(default=True), max_auto_requests=100000,
        invariance_max_flip_rate=GATE_RATE if gate else None)

    if engine == "jev":
        log_spend(str(date.today()), f"{arm} seed {seed}: steering round (sent)",
                  N_LABELS, N_LABELS * (2 if gate else 1), counter.requests,
                  f"decision={outcome.decision}; {counter.input_tokens:,} input / "
                  f"{counter.output_tokens:,} output tokens so far")

    proposals = proposal_rows(arm, engine, seed, outcome, gate)
    write_rows(proposals, PROPOSALS_OUT)

    metrics, serve_sent = score_held_out(ws, arm, engine, seed, factory)
    write_rows([metrics.as_row()], METRICS_OUT)
    print(f"  {arm} seed {seed}: {outcome.decision}, v{ws.version}, "
          f"accuracy {metrics.accuracy:.4f}, flip {metrics.flip_rate:.4f}")

    total_sent = (counter.requests if engine == "jev" else 0)
    return {"arm": arm, "seed": seed, "engine": engine, "decision": outcome.decision,
            "version": ws.version, "total_jev_requests": total_sent,
            "input_tokens": counter.input_tokens if counter else 0,
            "output_tokens": counter.output_tokens if counter else 0}, ws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=list(ARM_CONFIG), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scratch", type=Path, default=Path("var/bios_steering"))
    parser.add_argument("--record", action="store_true", default=True)
    parser.add_argument("--no-record", dest="record", action="store_false")
    args = parser.parse_args()
    load_dotenv()

    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not SPEND_LOG.exists():
        SPEND_LOG.write_text("| when | step | items | requests priced | requests sent | notes |\n"
                             "|---|---|---:|---:|---:|---|\n")

    for seed in args.seeds:
        workspace_root = args.scratch / f"{args.arm}-seed{seed}"
        summary, ws = one_run(args.arm, seed, workspace_root)
        print(json.dumps(summary))
        if args.record:
            out = RECORDINGS / f"{args.arm}-seed{seed}"
            record(ws, SCORE_NAME, out, FIXTURES,
                  title=f"bios_gender {args.arm}, seed {seed}",
                  provenance=(f"Simulated labeler (the corpus's own occupation label), 140 "
                              f"labels, one steering round "
                              f"({'with' if ARM_CONFIG[args.arm]['gate'] else 'without'} the "
                              f"gender-invariance gate), {ARM_CONFIG[args.arm]['engine']} "
                              f"engine. studies/PREREGISTERED.md, 'does the engine read "
                              f"gender, and can the layer refuse to?'"))
            print(f"  recorded to {out}")


if __name__ == "__main__":
    main()
