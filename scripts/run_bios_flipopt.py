#!/usr/bin/env python
"""L3/L4/L5/L6 and the twin-averaging baseline: optimising the head against the flip.

    python scripts/run_bios_flipopt.py --arm baseline --seeds 1 2
    python scripts/run_bios_flipopt.py --arm L4 --seeds 1 2

``studies/PREREGISTERED.md``'s "optimising the head against the flip" section. Every arm
starts from the same place: the 140 labels ``fixtures/bios/recordings/L1-seed<N>`` recorded
(read directly from that recording's ``feedback.jsonl`` -- this script never reruns the
labeling loop itself), refit once on the scorecard those 140 labels were labeled against
(``fixtures/bios/scorecards/v1.yaml``, one feature: the engine's own holistic answer). That
is the same state ``run_bios_steering.py``'s L1/L2 arms were in immediately *before* their own
steering round; from there:

* **baseline** does not refit at all. It replays the seed's *actual* L1 recording (labels,
  every ordinary refit, and L1's own steering round) to get L1's final fitted head, then scores
  P(surgeon) as the mean of that head's probability on a held-out bio and on its counterfactual
  twin.
* **L3**, **L4** refit the same single-feature scorecard differently (twin-augmented rows;
  the invariance-penalty loss) and never steer.
* **L5** first drops the holistic feature if it fails the 2% gate on the labeled twins (it
  does, on every seed so far), then runs one steering round with the L2 gate on, same as L2.
* **L6** runs one steering round whose mismatch set is the labeled items that flip under
  ``swap_gender``, shown alongside the ordinary reviewer disagreements, gated the same way.

Laya only (free, local): ``studies/PREREGISTERED.md``'s rule for this section. No Jev calls
are made and none are logged to any spend file. The 140 labeled items' twins are answered once
per seed and cached under ``var/bios_flipopt/`` (``--offline`` skips answering anything not
already cached there, for a replay with no GPU).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_gender import Verdict, score_arm, write_rows  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.counterfactual import swap_gender  # noqa: E402
from jev_flywheel.fit import (  # noqa: E402
    build_training_set, fit_head, fit_head_invariance, with_fit)
from jev_flywheel.invariance import (  # noqa: E402
    DEFAULT_MAX_FLIP_RATE, elements_over_gate, flip_rate, flip_rates_by_element)
from jev_flywheel.items import FeedbackItem, Item, JsonlStore  # noqa: E402
from jev_flywheel.jev import JevSession  # noqa: E402
from jev_flywheel.ladder import tier_for  # noqa: E402
from jev_flywheel.loop import refit  # noqa: E402
from jev_flywheel.proposal import ProposalError, parse_proposal  # noqa: E402
from jev_flywheel.recording import replay  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.steer import ScriptedApprover, run_steering  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

FIXTURES = Path("fixtures/bios")
RECORDINGS = FIXTURES / "recordings"
SCORE_NAME = "Occupation"
N_LABELS = 140
GATE_RATE = DEFAULT_MAX_FLIP_RATE   # 0.02, fixed in the pre-registration
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
# Exploratory only (2026-09-22, requested after seeing the seed-1 L3/L4/baseline rows): does
# any lambda past the pre-registered grid trade accuracy for fewer flips, or does the "only
# the intercept moves which pairs straddle the boundary" argument hold all the way? Swept and
# reported, tagged "exploratory": true; never eligible to become the operating point --
# fit_head_invariance's ``registered=LAMBDA_GRID`` keeps that choice on the registered grid only.
EXPLORATORY_LAMBDAS = (1000.0, 10000.0)
CACHE_DIR = Path("var/bios_flipopt")
METRICS_OUT = Path("studies/bios_flipopt.jsonl")
PROPOSALS_OUT = Path("studies/bios_flipopt_proposals.jsonl")
LAMBDA_CURVE_OUT = Path("studies/bios_flipopt_lambda_curve.jsonl")
THRESHOLD_OUT = Path("studies/bios_flipopt_threshold.jsonl")
GATE_SENSITIVITY_OUT = Path("studies/bios_flipopt_gate_sensitivity.jsonl")

ARMS = ("baseline", "L3", "L4", "L5", "L6")

_LAYA_CLIENT = None


def laya_client():
    """One warm Laya client for the whole run -- see run_bios_steering.py's identical helper."""
    global _LAYA_CLIENT
    if _LAYA_CLIENT is None:
        from jev_flywheel.laya import LayaClient

        _LAYA_CLIENT = LayaClient()
        _LAYA_CLIENT.warm()
    return _LAYA_CLIENT


# ---- the shared starting point: 140 labels, one feature, refit once -------------------------

def base_workspace(seed: int, scratch: Path) -> Workspace:
    """The state every arm starts from: the L1 recording's 140 labels, replayed as plain
    feedback (no steering) against the v1 scorecard, refit once.

    This reproduces exactly the state L1's own steering round ran against -- the same labels,
    the same single-feature fit -- without importing anything L1 proposed, so L3/L4/L5/L6 are
    each their own answer to "what should happen after 140 labels", not a modification of L1's.
    """
    ws = Workspace.init(scratch, FIXTURES, force=True, answers="answers-laya.jsonl.gz",
                        engine="laya")
    feedback_path = RECORDINGS / f"L1-seed{seed}" / "feedback.jsonl"
    if not feedback_path.exists():
        raise SystemExit(f"no L1 recording for seed {seed} at {feedback_path} -- run "
                         f"scripts/run_bios_steering.py --arm L1 --seeds {seed} first "
                         "(the other agent's arm; this script only reads its feedback.jsonl).")
    for record in JsonlStore(feedback_path, FeedbackItem):
        ws.add_feedback(record)
    assert ws.n_labeled(SCORE_NAME) == N_LABELS, ws.n_labeled(SCORE_NAME)
    outcome = refit(ws, SCORE_NAME)
    if not outcome.result or not outcome.result.fitted:
        raise SystemExit(f"seed {seed}: the base refit did not fit ({outcome.reasons})")
    return ws


# ---- the 140 labeled items' twins: answered once per seed, cached locally -------------------

def labeled_item_ids(ws: Workspace) -> List[str]:
    from jev_flywheel.fit import latest_feedback

    return list(latest_feedback(ws.feedback(), SCORE_NAME))


def twin_cache_path(seed: int) -> Path:
    return CACHE_DIR / f"twins-seed{seed}.jsonl"


def build_twin_cache(ws: Workspace, seed: int, item_ids: List[str], question: Mapping[str, Any],
                     offline: bool = False) -> AnswerCache:
    """Laya's answer to the (single, holistic) question on every labeled item's gender-swapped
    twin, cached under ``var/bios_flipopt/`` -- our own cache, never the workspace's, since
    these items are not real corpus items.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = AnswerCache(twin_cache_path(seed))
    twins = [Item(id=f"{item_id}__twin", text=swap_gender(ws.item(item_id).text).text)
            for item_id in item_ids]
    questions = {SCORE_NAME: dict(question)}
    plan = cache.plan([t.id for t in twins], questions)
    if plan.requests and offline:
        raise SystemExit(f"seed {seed}: {plan.requests} twin answers are not cached and "
                         "--offline was given; run once without --offline first.")
    if plan.requests:
        session = JevSession(client_factory=laya_client)
        report = asyncio.run(cache.fill(session, twins, questions))
        print(f"  seed {seed}: answered {report.requested} labeled-item twins on Laya "
             f"({report.failures} failures)")
    return cache


def twin_feature_rows(score, cache: AnswerCache, questions: Mapping[str, Any],
                      item_ids: List[str]) -> Dict[str, Dict[str, float]]:
    rows = {}
    for item_id in item_ids:
        answers = cache.answers_for(f"{item_id}__twin", questions)
        if answers is not None:
            rows[item_id] = score.feature_vector(answers)
    return rows


# ---- the mechanism: where does each arm's threshold sit? -----------------------------------

def holistic_feature_name(features: List[str]) -> Optional[str]:
    return next((f for f in features if "holistic" in f), None)


def threshold_row(arm: str, seed: int, card: Scorecard,
                  positive: str = "surgeon") -> Dict[str, Any]:
    """The intercept and the holistic feature's weight of a fitted scorecard, so a reader can
    see whether an arm moved the *decision boundary* (raw Laya's own threshold is 0, in the
    holistic feature's own units -- ``fixtures/bios/scorecards/v1.yaml``'s intercept 0.0,
    weight 2.0) rather than just shrinking the weight on it.

    A flip is an item and its twin landing on opposite sides of ``weight * x + intercept == 0``.
    Scaling ``weight`` down moves the boundary in x-space to ``-intercept / weight`` -- which
    can be a *large* move numerically -- but it does not by itself change *which* items sit on
    which side unless ``intercept`` also moves relative to how spread out ``x`` is; an L2 (or
    L2 + invariance-penalty) fit's log-loss term pins ``intercept`` to match the label's base
    rate, which barely moves across the lambda grid here. That is the mechanism note requested
    after the seed-1 L3/L4/baseline rows: see the arm's row here next to its lambda curve.
    """
    from jev_flywheel.models import class_weights

    score = card.score(SCORE_NAME)
    decision = score.decision
    weights = class_weights(decision.head()).get(positive, {})
    intercept = float(weights.get("intercept", 0.0))
    hf = holistic_feature_name(decision.features)
    holistic_weight = float(weights.get(hf, 0.0)) if hf else None
    boundary = (-intercept / holistic_weight) if holistic_weight else None
    return {
        "arm": arm, "seed": seed, "positive_class": positive, "model": decision.model,
        "features": list(decision.features), "intercept": round(intercept, 4),
        "holistic_feature": hf,
        "holistic_weight": None if holistic_weight is None else round(holistic_weight, 4),
        "decision_boundary_in_holistic_units": None if boundary is None else round(boundary, 4),
        "raw_laya_boundary": 0.0,
    }


# ---- held-out scoring -------------------------------------------------------------------

def score_held_out(ws: Workspace, arm: str, seed: int, scorecard: Optional[Scorecard] = None,
                   offline: bool = False,
                   average_with_twin: bool = False) -> "object":
    """Score a scorecard (``ws``'s current version, or ``scorecard`` if given) on the 2,000
    held-out bios and their counterfactual twins, all answered by Laya.

    ``average_with_twin`` is the baseline arm: P(surgeon) becomes the mean of the item's own
    probability and its twin's, for both the item and the twin, which makes the two identical
    and the flip rate zero by construction.
    """
    card = scorecard or ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    test_items = ws.split("test")
    twin_items = ws.split("counterfactual")
    all_items = test_items + twin_items
    plan = ws.cache.plan([i.id for i in all_items], questions)
    if plan.requests and offline:
        raise SystemExit(f"{arm} seed {seed}: {plan.requests} held-out answers are missing "
                         "and --offline was given.")
    if plan.requests:
        session = JevSession(client_factory=laya_client)
        report = asyncio.run(ws.cache.fill(session, all_items, questions))
        print(f"  {arm} seed {seed}: answered {report.requested} held-out items on Laya "
             f"({report.failures} failures)")

    def p_surgeon(item) -> float:
        answers = ws.cache.partial_answers_for(item.id, questions)
        result = predict(score, answers)
        confidence = result.confidence if result.confidence is not None else 0.5
        return confidence if result.value == "surgeon" else 1.0 - confidence

    twin_of = {t.metadata["counterfactual_of"]: t for t in twin_items}

    def verdict_for(item, p: float) -> Verdict:
        predicted = "surgeon" if p >= 0.5 else "physician"
        return Verdict(item.id, predicted, p, item.reference_label, item.metadata.get("gender"))

    verdicts: List[Verdict] = []
    twins: Dict[str, Verdict] = {}
    for item in test_items:
        p_item = p_surgeon(item)
        twin = twin_of.get(item.id)
        p_twin = p_surgeon(twin) if twin is not None else None
        if average_with_twin and p_twin is not None:
            p_item = p_twin = (p_item + p_twin) / 2.0
        verdicts.append(verdict_for(item, p_item))
        if twin is not None and p_twin is not None:
            twins[item.id] = verdict_for(twin, p_twin)

    return score_arm(arm=arm, engine="laya", verdicts=verdicts, twins=twins, seed=seed,
                     version=getattr(card, "version", None), n_labels=N_LABELS, redacted=True)


# ---- proposal bookkeeping (L5/L6 only; adapted from run_bios_steering.py's pattern) ----------

def wire_flip_rate(flip_rates: Optional[Dict[str, float]], key: str) -> Optional[float]:
    if not flip_rates:
        return None
    for wire, rate in flip_rates.items():
        if wire == key or wire.endswith(f".{key}"):
            return rate
    return None


def proposal_rows(arm: str, seed: int, outcome) -> List[Dict[str, Any]]:
    if not outcome.analyst_reply:
        return []
    try:
        proposal = parse_proposal(outcome.analyst_reply)
    except ProposalError:
        return [{"arm": arm, "engine": "laya", "seed": seed, "key": None,
                 "wording": outcome.analyst_reply, "passed_fit": None,
                 "flip_rate_on_labeled": None, "passed_gate": None, "promoted": False,
                 "decision": outcome.decision, "note": "unparseable analyst reply"}]
    reasons_text = str(outcome.detail.get("reasons") or "")
    non_gate_reason_present = ("Brier improved by" in reasons_text
                               or "accuracy fell by" in reasons_text)
    rows = []
    for added in proposal.add:
        flip = wire_flip_rate(outcome.invariance_flip_rates, added.key)
        passed_gate = (flip <= GATE_RATE) if flip is not None else None
        if outcome.decision == "promoted":
            passed_fit = True
        elif outcome.decision == "rejected_by_metrics":
            passed_fit = not non_gate_reason_present
        else:
            passed_fit = None
        rows.append({
            "arm": arm, "engine": "laya", "seed": seed, "key": added.key,
            "wording": added.instructions, "question_type": added.question_type,
            "passed_fit": passed_fit, "flip_rate_on_labeled": flip,
            "passed_gate": passed_gate, "promoted": outcome.decision == "promoted",
            "decision": outcome.decision, "root_cause": proposal.root_cause})
    if not proposal.add:
        rows.append({
            "arm": arm, "engine": "laya", "seed": seed, "key": None, "wording": None,
            "passed_fit": None, "flip_rate_on_labeled": None, "passed_gate": None,
            "promoted": False, "decision": outcome.decision, "root_cause": proposal.root_cause,
            "note": "no element proposed" if outcome.decision != "invalid_proposal"
                    else "invalid proposal"})
    return rows


# ---- the arms ---------------------------------------------------------------------------

def run_baseline(seed: int, scratch: Path, offline: bool) -> Tuple[Any, Workspace, Dict[str, Any]]:
    """Replay the seed's actual L1 recording (offline: engine matches, so its own extra
    answers are restored and no network or GPU call is needed beyond serving the held-out
    set), then score P(surgeon) as the item/twin average of L1's final fitted head."""
    ws = replay(RECORDINGS / f"L1-seed{seed}", scratch, FIXTURES,
               answers="answers-laya.jsonl.gz", engine="laya",
               client_factory=None if offline else laya_client)
    metrics = score_held_out(ws, "baseline", seed, offline=offline, average_with_twin=True)
    threshold = threshold_row("baseline", seed, ws.scorecard())
    return metrics, ws, threshold


def run_l3(seed: int, scratch: Path, offline: bool) -> Tuple[Any, Workspace, Dict[str, Any]]:
    ws = base_workspace(seed, scratch)
    item_ids = labeled_item_ids(ws)
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    question = questions[SCORE_NAME]
    twins = build_twin_cache(ws, seed, item_ids, question, offline=offline)
    rows = twin_feature_rows(score, twins, questions, item_ids)

    training = build_training_set(score, questions, ws.cache, ws.feedback(), twin_rows=rows)
    result = fit_head(training, score)
    if not result.fitted:
        raise SystemExit(f"L3 seed {seed} did not fit: {result.reason}")
    fitted_card = with_fit(card, SCORE_NAME, result)
    ws.commit_scorecard(fitted_card, kind="fit", provenance={"arm": "L3", "seed": seed})
    metrics = score_held_out(ws, "L3", seed, offline=offline)
    threshold = threshold_row("L3", seed, fitted_card)
    return metrics, ws, threshold


def run_l4(seed: int, scratch: Path, offline: bool) -> Tuple[Any, Workspace, List[Dict], Dict[str, Any]]:
    ws = base_workspace(seed, scratch)
    item_ids = labeled_item_ids(ws)
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    question = questions[SCORE_NAME]
    twins = build_twin_cache(ws, seed, item_ids, question, offline=offline)
    rows = twin_feature_rows(score, twins, questions, item_ids)

    training = build_training_set(score, questions, ws.cache, ws.feedback())
    # Sweeps the pre-registered grid plus an exploratory extension past it (requested
    # 2026-09-22, after the seed-1 L3/L4/baseline rows): every lambda is swept and reported,
    # but ``registered=LAMBDA_GRID`` keeps the operating point on the pre-registered grid only.
    result = fit_head_invariance(training, score, rows, lambdas=LAMBDA_GRID + EXPLORATORY_LAMBDAS,
                                 registered=LAMBDA_GRID)

    from jev_flywheel.fit import FitResult

    wrapped = FitResult(status="fitted", tier=result.tier, n=result.n,
                        n_effective=result.n_effective, head=result.head,
                        calibration={"method": "none"}, provenance=result.provenance)
    fitted_card = with_fit(card, SCORE_NAME, wrapped)
    ws.commit_scorecard(fitted_card, kind="fit",
                        provenance={"arm": "L4", "seed": seed,
                                   "operating_lambda": result.operating_lambda,
                                   "chosen_c": result.chosen_c})
    metrics = score_held_out(ws, "L4", seed, offline=offline)
    curve_rows = [{
        "arm": "L4", "seed": seed, "lambda": point.lam, "chosen_c": result.chosen_c,
        "oof_accuracy": round(point.oof_accuracy, 4), "oof_log_loss": round(point.oof_log_loss, 4),
        "oof_mean_abs_dp_labeled_twins": round(point.oof_mean_abs_dp, 4),
        "intercept": round(point.intercept, 4),
        "holistic_weight": round(next(iter(point.weights.values())), 4) if point.weights else None,
        "operating_point": point.lam == result.operating_lambda,
        "exploratory": point.lam not in LAMBDA_GRID and point.lam != 0.0,
    } for point in result.points]
    threshold = threshold_row("L4", seed, fitted_card)
    return metrics, ws, curve_rows, threshold


def run_l5(seed: int, scratch: Path, offline: bool, gate_rate: float = GATE_RATE,
          exploratory: bool = False) -> Tuple[Any, Workspace, List[Dict[str, Any]], Dict[str, Any]]:
    """``gate_rate``/``exploratory`` (added 2026-09-22, after seeing every L2 seed reject every
    proposal at the registered 2% gate -- evidence questions flip 4-7% on the labeled twins,
    courtesy-title questions 14-24%): the registered run is ``gate_rate=GATE_RATE,
    exploratory=False``; ``main()`` additionally runs 5% and 10% as an explicitly exploratory
    extension, outside the pre-registration's own 2% rule, so a reader can see the trade-off
    when the least gender-sensitive evidence questions are let back in.
    """
    ws = base_workspace(seed, scratch)
    item_ids = labeled_item_ids(ws)
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    question = questions[SCORE_NAME]
    twins = build_twin_cache(ws, seed, item_ids, question, offline=offline)

    # ``before``/``after`` are keyed by the *question* name (there is exactly one existing
    # question at this stage, the holistic answer itself: score.decision.features is derived
    # from it by score.feature_vector, but the gate is asked of the answer, not the derived
    # number). ``after`` is rekeyed from "<id>__twin" back to "<id>" so the two line up.
    before = {i: ws.cache.partial_answers_for(i, questions) for i in item_ids}
    after = {i: twins.partial_answers_for(f"{i}__twin", questions) for i in item_ids}
    rates = flip_rates_by_element(before, after, [SCORE_NAME])
    dropped = elements_over_gate(rates, max_flip_rate=gate_rate)
    print(f"  L5 seed {seed} (gate {gate_rate:.0%}): existing-element flip rates {rates}, "
         f"dropped {dropped}")

    # The only existing question at this stage is the holistic answer, so "gate every element"
    # reduces to "drop every feature derived from it, if it fails" -- there is nothing else to
    # gate yet (L1/L2's promoted elements are never in this arm's starting scorecard; see
    # base_workspace's docstring).
    surviving = [] if SCORE_NAME in dropped else list(score.decision.features)
    training = build_training_set(score, questions, ws.cache, ws.feedback())
    from jev_flywheel.fit import FitResult
    from collections import Counter

    if surviving:
        result = fit_head(training, score)
        gated_card = with_fit(card, SCORE_NAME, result) if result.fitted else card
        note = "holistic answer passed the gate" if surviving == score.decision.features \
            else "partially gated"
    else:
        counts = Counter(training.labels)
        majority = max(counts, key=counts.get)
        p_majority = counts[majority] / sum(counts.values())
        import math

        logit = math.log(p_majority / (1 - p_majority)) if 0 < p_majority < 1 else 0.0
        head = {"model": "multinomial_logistic", "classes": list(score.decision.classes),
               "weights": {majority: {"intercept": logit}}}
        wrapped = FitResult(status="fitted", tier=tier_for(training.n_effective), n=training.n,
                            n_effective=training.n_effective, head=head,
                            calibration={"method": "none"},
                            provenance={"arm": "L5", "seed": seed, "note": "majority class: "
                                       "every existing feature failed the invariance gate"})
        gated_card = with_fit(card, SCORE_NAME, wrapped)
        note = "nothing survived the gate; scored the majority class"
    ws.commit_scorecard(gated_card, kind="fit",
                        provenance={"arm": "L5", "seed": seed, "gate_rate": gate_rate,
                                   "dropped_elements": dropped, "note": note})

    # The registered run keeps the plain "L5" arm name (everything downstream, e.g. the
    # Outcome table, reads it that way); an exploratory gate gets its own name so it never
    # mixes into "L5"'s row in studies/bios_flipopt.jsonl -- the gate rate itself lives in
    # the threshold/proposal rows, which do have room for it.
    arm_label = "L5" if not exploratory else f"L5-gate{gate_rate:.0%}".replace("%", "")
    outcome = run_steering(
        ws, SCORE_NAME, allow_spend=True, client_factory=laya_client,
        hitl_handler=ScriptedApprover(default=True), max_auto_requests=100000,
        invariance_max_flip_rate=gate_rate)
    rows = proposal_rows(arm_label, seed, outcome)
    surviving_after_steering = list(ws.scorecard().score(SCORE_NAME).decision.features)
    for row in rows:
        row["dropped_elements"] = dropped
        row["gate_rate"] = gate_rate
        row["exploratory"] = exploratory
    metrics = score_held_out(ws, arm_label, seed, offline=offline)
    threshold = threshold_row("L5", seed, ws.scorecard())
    threshold.update({"gate_rate": gate_rate, "exploratory": exploratory,
                      "dropped_elements": dropped,
                      "surviving_elements_after_steering": surviving_after_steering,
                      "steering_decision": outcome.decision})
    return metrics, ws, rows, threshold


def flip_mismatch_entries(ws: Workspace, item_ids: List[str], twins: AnswerCache,
                          score, questions) -> List[Dict[str, Any]]:
    """L6's mismatch set: labeled items whose verdict differs under ``swap_gender``, in the
    shape ``FlywheelHost``'s ordinary mismatches take (see ``host.py``'s ``_feedback_examples``)."""
    entries = []
    question = questions[SCORE_NAME]
    for item_id in item_ids:
        item_answers = ws.cache.partial_answers_for(item_id, questions)
        twin_answers = twins.partial_answers_for(f"{item_id}__twin", questions)
        if SCORE_NAME not in item_answers or SCORE_NAME not in twin_answers:
            continue
        item_result = predict(score, item_answers)
        twin_result = predict(score, twin_answers)
        if item_result.value == twin_result.value:
            continue
        entries.append({
            "item_id": item_id, "text": ws.item(item_id).text,
            "we_said": item_result.value,
            "confidence_when_shown": round(item_result.confidence or 0.0, 4),
            "human_said": twin_result.value,
            "human_comment": "only the pronouns differ",
            "element_answers": {},
            "top_drivers_now": [],
        })
    return entries


def run_l6(seed: int, scratch: Path, offline: bool, gate_rate: float = GATE_RATE,
          exploratory: bool = False) -> Tuple[Any, Workspace, List[Dict[str, Any]], Dict[str, Any]]:
    """``gate_rate``/``exploratory``: same meaning as ``run_l5``'s. The registered run is
    ``gate_rate=GATE_RATE``; ``main()`` reruns at 5%/10% only if the registered run's gate
    empties the promoted-element set (nothing promoted)."""
    ws = base_workspace(seed, scratch)
    item_ids = labeled_item_ids(ws)
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    question = questions[SCORE_NAME]
    twins = build_twin_cache(ws, seed, item_ids, question, offline=offline)
    mismatches = flip_mismatch_entries(ws, item_ids, twins, score, questions)
    print(f"  L6 seed {seed} (gate {gate_rate:.0%}): {len(mismatches)} of {len(item_ids)} "
         "labeled items flip under the gender swap; shown to the analyst alongside the "
         "ordinary disagreements")

    arm_label = "L6" if not exploratory else f"L6-gate{gate_rate:.0%}".replace("%", "")
    outcome = run_steering(
        ws, SCORE_NAME, allow_spend=True, client_factory=laya_client,
        hitl_handler=ScriptedApprover(default=True), max_auto_requests=100000,
        invariance_max_flip_rate=gate_rate, flip_mismatches=mismatches)
    rows = proposal_rows(arm_label, seed, outcome)
    for row in rows:
        row["n_flip_mismatches_shown"] = len(mismatches)
        row["gate_rate"] = gate_rate
        row["exploratory"] = exploratory
    metrics = score_held_out(ws, arm_label, seed, offline=offline)
    threshold = threshold_row("L6", seed, ws.scorecard())
    threshold.update({"gate_rate": gate_rate, "exploratory": exploratory,
                      "steering_decision": outcome.decision,
                      "n_flip_mismatches_shown": len(mismatches)})
    return metrics, ws, rows, threshold


EXPLORATORY_GATES = (0.05, 0.10)


def one_run(arm: str, seed: int, scratch: Path, offline: bool,
           exploratory_gates: bool = False) -> None:
    if arm == "baseline":
        metrics, _, threshold = run_baseline(seed, scratch, offline)
        write_rows([metrics.as_row()], METRICS_OUT)
        write_rows([threshold], THRESHOLD_OUT)
    elif arm == "L3":
        metrics, _, threshold = run_l3(seed, scratch, offline)
        write_rows([metrics.as_row()], METRICS_OUT)
        write_rows([threshold], THRESHOLD_OUT)
    elif arm == "L4":
        metrics, _, curve, threshold = run_l4(seed, scratch, offline)
        write_rows([metrics.as_row()], METRICS_OUT)
        write_rows(curve, LAMBDA_CURVE_OUT)
        write_rows([threshold], THRESHOLD_OUT)
    elif arm == "L5":
        metrics, _, proposals, threshold = run_l5(seed, scratch, offline)
        write_rows([metrics.as_row()], METRICS_OUT)
        write_rows(proposals, PROPOSALS_OUT)
        write_rows([threshold], THRESHOLD_OUT)
        # Every registered L5 seed drops to (or towards) the majority class at the 2% gate --
        # the pre-registration's own worst case. Always run the exploratory 5%/10% extension
        # (requested 2026-09-22) so a reader sees the trade-off when the least
        # gender-sensitive evidence questions are let back in.
        if exploratory_gates:
            for gate in EXPLORATORY_GATES:
                em, _, eprop, ethresh = run_l5(seed, scratch / f"gate{gate:.0%}", offline,
                                              gate_rate=gate, exploratory=True)
                write_rows([em.as_row()], METRICS_OUT)
                write_rows(eprop, PROPOSALS_OUT)
                write_rows([ethresh], GATE_SENSITIVITY_OUT)
    elif arm == "L6":
        metrics, _, proposals, threshold = run_l6(seed, scratch, offline)
        write_rows([metrics.as_row()], METRICS_OUT)
        write_rows(proposals, PROPOSALS_OUT)
        write_rows([threshold], THRESHOLD_OUT)
        # Same extension, but only if the registered 2% gate emptied the promoted set (nothing
        # promoted) -- L6 starts from a richer mismatch set than L5, so it is not guaranteed to.
        if exploratory_gates and not threshold.get("steering_decision") == "promoted":
            for gate in EXPLORATORY_GATES:
                em, _, eprop, ethresh = run_l6(seed, scratch / f"gate{gate:.0%}", offline,
                                              gate_rate=gate, exploratory=True)
                write_rows([em.as_row()], METRICS_OUT)
                write_rows(eprop, PROPOSALS_OUT)
                write_rows([ethresh], GATE_SENSITIVITY_OUT)
    else:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {ARMS}")
    print(f"{arm} seed {seed}: accuracy {metrics.accuracy:.4f}, flip {metrics.flip_rate:.4f}, "
         f"ece {metrics.ece:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scratch", type=Path, default=Path("var/bios_flipopt"))
    parser.add_argument("--offline", action="store_true",
                        help="never call Laya; fail if an answer this run needs is not "
                             "already cached (in var/bios_flipopt/ or fixtures/bios/).")
    parser.add_argument("--exploratory-gates", action="store_true",
                        help="L5/L6 only: also run the 5%%/10%% exploratory gate extension "
                             "(studies/bios_flipopt_gate_sensitivity.jsonl).")
    args = parser.parse_args()
    load_dotenv()

    for seed in args.seeds:
        scratch = args.scratch / f"{args.arm}-seed{seed}" / "var"
        one_run(args.arm, seed, scratch, args.offline, args.exploratory_gates)


if __name__ == "__main__":
    main()
