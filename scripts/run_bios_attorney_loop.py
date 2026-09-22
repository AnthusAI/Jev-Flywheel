#!/usr/bin/env python
""""The learning loop on the pair that matters": J0/L0, J1/J2/L1/L2 (seeds 1-3), the
twin-averaging baseline, and the shortlist re-run, on the paralegal/attorney pair
(``studies/PREREGISTERED.md``, final section).

    python scripts/run_bios_attorney_loop.py --arm J0
    python scripts/run_bios_attorney_loop.py --arm L0
    python scripts/run_bios_attorney_loop.py --arm J1 --seeds 1 2 3
    python scripts/run_bios_attorney_loop.py --arm J2 --seeds 1 2 3
    python scripts/run_bios_attorney_loop.py --arm L1 --seeds 1 2 3
    python scripts/run_bios_attorney_loop.py --arm L2 --seeds 1 2 3
    python scripts/run_bios_attorney_loop.py --baseline-from L1 --baseline-seed 1

Follows ``scripts/run_bios_steering.py``'s pattern (140 labels from the simulated labeler, one
steering round, the invariance gate on for J2/L2) exactly, pointed at
``fixtures/bios_attorney`` instead of ``fixtures/bios``, with "attorney" (not "surgeon") as the
positive class -- so it reuses ``scripts/bios_pairs.py``'s pair-generic metric functions rather
than ``scripts/bios_gender.py``'s surgeon-specific ones. J0/L0 read the engine's own answer to
the v1 question directly from the fixtures, no workspace and no labels, exactly like
``scripts/run_bios_pairs.py``.

Every arm's held-out scores also feed the shortlist measurement
(``scripts/bios_attorney_shortlist.py``): a ranked screen over the same 1,000 real attorney +
1,000 paralegal held-out bios, four-fifths ratio and the counterfactual counts at cuts 250/500/
1,000. Rows go to ``studies/bios_attorney.jsonl`` (metrics), ``studies/bios_attorney_proposals.jsonl``
(every proposed element's wording) and ``studies/bios_attorney_shortlist.jsonl`` (the screen).

Money (the pre-registration's rule, this section): 4,000 pool + 2,000 held-out-twin requests
(actually smaller here -- see ``studies/bios_attorney_spend.md`` and the fixture builder's
deviation note: the paralegal pool is exhausted at 146, not 2,000) plus J1/J2's steering
top-ups (<=140 + <=140 per proposal, 4,000 to serve a promoted element), hard cap 24,000 for
this section.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bios_attorney_shortlist import shortlist_metrics  # noqa: E402
from bios_gender import Verdict, write_rows  # noqa: E402
from bios_pairs import score_pair  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from jev_flywheel.head import decide  # noqa: E402
from jev_flywheel.jev import JevSession, _default_client  # noqa: E402
from jev_flywheel.proposal import ProposalError, parse_proposal  # noqa: E402
from jev_flywheel.recording import record  # noqa: E402
from jev_flywheel.scoring import predict  # noqa: E402
from jev_flywheel.simulate import label_with_reference  # noqa: E402
from jev_flywheel.steer import ScriptedApprover, run_steering  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

FIXTURES = Path("fixtures/bios_attorney")
RECORDINGS = FIXTURES / "recordings"
N_LABELS = 140
GATE_RATE = 0.02
SCORE_NAME = "Occupation"
PAIR = "paralegal_attorney"
POSITIVE = "attorney"      # less-female label; scorecard's positive class
NEGATIVE = "paralegal"     # more-female label

ARM_CONFIG = {
    "J1": {"engine": "jev", "gate": False},
    "J2": {"engine": "jev", "gate": True},
    "L1": {"engine": "laya", "gate": False},
    "L2": {"engine": "laya", "gate": True},
}
ANSWER_FILES = {"jev": "answers.jsonl.gz", "laya": "answers-laya.jsonl.gz"}
SPEND_LOG = Path("studies/bios_attorney_spend.md")
PROPOSALS_OUT = Path("studies/bios_attorney_proposals.jsonl")
METRICS_OUT = Path("studies/bios_attorney.jsonl")
SHORTLIST_OUT = Path("studies/bios_attorney_shortlist.jsonl")
ELEMENTS_OUT = Path("studies/bios_attorney_elements.jsonl")
HOLISTIC_FEATURE = "self.holistic.clr.attorney"
V1_HOLISTIC_WEIGHT_PARALEGAL = -2.0  # v1's linear_threshold weight (2.0, positive_class=attorney)
                                     # re-expressed in the multinomial head's paralegal-logit sign


class CountingClient:
    """Same pattern as ``run_bios_steering.CountingClient``: wraps a real client and counts
    every request/token a steering round (and its top-ups) actually sends."""

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
    global _LAYA_CLIENT
    if _LAYA_CLIENT is None:
        from jev_flywheel.laya import LayaClient

        _LAYA_CLIENT = LayaClient()
        _LAYA_CLIENT.warm()
    return _LAYA_CLIENT


def client_factory_for(engine: str, counter: Optional[CountingClient] = None):
    if engine == "laya":
        return laya_client
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
    """Copied from ``run_bios_steering.proposal_rows`` verbatim (same derivation rules, same
    reasons the direct-computation approach is used instead of parsing prose)."""
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


def load_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_held_out_items() -> Dict[str, dict]:
    return {row["id"]: row for row in load_jsonl(FIXTURES / "items.jsonl")
            if row["metadata"]["split"] in ("test", "counterfactual")}


def held_out_split() -> Tuple[List[str], Dict[str, str]]:
    """(test ids, {orig_id: twin_id})."""
    items = load_held_out_items()
    test_ids = [i for i, r in items.items() if r["metadata"]["split"] == "test"]
    twin_of = {r["metadata"]["counterfactual_of"]: i for i, r in items.items()
               if r["metadata"]["split"] == "counterfactual"}
    return test_ids, twin_of


def run_shortlist(arm: str, engine: str, seed: Optional[int],
                  scores_by_id: Dict[str, float]) -> None:
    """``scores_by_id`` must cover every held-out test item (both labels) and every twin id."""
    items_meta = load_held_out_items()
    test_ids, twin_of = held_out_split()
    items = {i: (items_meta[i]["metadata"]["reference_label"], items_meta[i]["metadata"]["gender"])
             for i in test_ids}
    scores = {i: scores_by_id[i] for i in test_ids}
    twin_scores = {orig: scores_by_id[twin_id] for orig, twin_id in twin_of.items()}
    rows = shortlist_metrics(items, scores, twin_scores, positive_label=POSITIVE)
    for row in rows:
        row.update({"arm": arm, "engine": engine, "seed": seed})
    write_rows(rows, SHORTLIST_OUT)
    print(f"  shortlist [{arm} seed {seed}]: "
          + ", ".join(f"top{r['cut']}={r['four_fifths_ratio']}" for r in rows))


def verdict_from_answer(item_id: str, meta: dict, answer: dict) -> Verdict:
    return Verdict(item_id, answer["choice"], answer["probabilities"][POSITIVE],
                   meta["reference_label"], meta["gender"])


def run_engine_alone(engine: str) -> None:
    """J0/L0: the engine's own v1 answer, no fitted head, no labels. Mirrors
    ``scripts/run_bios_pairs.py``'s ``score_from_fixtures`` exactly, renamed to this study's
    ``arm``/output files."""
    arm = "J0" if engine == "jev" else "L0"
    items = load_held_out_items()
    answers_path = FIXTURES / ANSWER_FILES[engine]
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} does not exist -- build it first")
    answers = {row["id"]: row["answers"]["Occupation"] for row in load_jsonl(answers_path)}
    test_ids, twin_of = held_out_split()
    missing = [i for i in test_ids if i not in answers] + [t for t in twin_of.values()
                                                            if t not in answers]
    if missing:
        raise SystemExit(f"{len(missing)} items have no {engine} answer, e.g. {missing[:3]}")

    verdicts = [verdict_from_answer(i, items[i]["metadata"], answers[i]) for i in test_ids]
    twins = {orig: verdict_from_answer(twin_id, items[twin_id]["metadata"], answers[twin_id])
             for orig, twin_id in twin_of.items()}
    metrics = score_pair(pair=PAIR, engine=engine, verdicts=verdicts, twins=twins)
    row = metrics.as_row()
    row.update({"arm": arm, "seed": None, "version": None, "n_labels": None})
    write_rows([row], METRICS_OUT)
    print(f"{arm}: accuracy {row['accuracy']}, flip {row['counterfactual_flip_rate']}")

    scores_by_id = dict(answers.items())
    scores_by_id = {i: a["probabilities"][POSITIVE] for i, a in answers.items()}
    run_shortlist(arm, engine, None, scores_by_id)


def label_prior_population(ws: Workspace) -> Optional[float]:
    """The fit's own recorded inverse-propensity estimate of the *pool's* attorney share, at
    the workspace's current scorecard version -- diagnosed in this section's Deviations block
    (2026-09-22): the fit calibrates its intercept to this pool prior, not to the held-out
    set's fixed 50/50 balance, which is why raw held-out accuracy can fall even as the fit's
    own out-of-fold metric improves. Exploratory addition, not a change to
    ``jev_flywheel/fit.py``: reads what the fit already recorded, in
    ``scorecards/lineage.jsonl``, for the workspace's current version. ``None`` if that
    version's provenance never fitted anything with a recorded prior (e.g. v1, before any
    refit) -- the caller skips the correction in that case.
    """
    lineage_path = ws.root / "scorecards" / "lineage.jsonl"
    if not lineage_path.exists():
        return None
    target = None
    for line in lineage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("version") == ws.version:
            target = row
    if target is None:
        return None
    prov = target.get("provenance") or {}
    fit_prov = prov.get("fit", prov)  # "steer"-kind nests the fit's own provenance under "fit"
    prior = fit_prov.get("label_prior_population")
    if not prior:
        return None
    return prior.get(POSITIVE)


def prior_corrected_accuracy(verdicts: List[Verdict], scores_by_id: Dict[str, float],
                             prior_attorney: Optional[float]
                            ) -> Tuple[Optional[float], Optional[float]]:
    """Exploratory (tagged as such wherever it is reported): re-threshold each held-out item's
    calibrated P(attorney) after shifting its logit by the log-odds difference between the
    held-out set's actual balance (50/50, by this section's own held-out sample) and the fit's
    own recorded population prior -- the accuracy the fitted head would have shown on the
    population it was actually calibrated for. Does not touch ``jev_flywheel/fit.py`` and does
    not change any ranking (the shift is a constant added to every item's logit), so the
    shortlist numbers, which depend only on ranking, are untouched by this correction.
    Returns ``(shift, corrected_accuracy)``, both ``None`` if there is no recorded prior to
    correct against.
    """
    if not prior_attorney or not (0.0 < prior_attorney < 1.0):
        return None, None
    shift = -math.log(prior_attorney / (1.0 - prior_attorney))  # logit(0.5) - logit(prior) == -logit(prior)
    correct = 0
    for v in verdicts:
        p = min(max(v.p_surgeon, 1e-9), 1 - 1e-9)
        logit = math.log(p / (1 - p)) + shift
        corrected_p = 1.0 / (1.0 + math.exp(-logit))
        predicted = POSITIVE if corrected_p >= 0.5 else NEGATIVE
        if predicted == v.truth:
            correct += 1
    return round(shift, 4), round(correct / len(verdicts), 4) if verdicts else None


def score_held_out(ws: Workspace, arm: str, engine: str, seed: int, factory
                   ) -> Tuple[dict, int, Dict[str, float]]:
    """Score the fitted (or unchanged) scorecard on the 2,000 held-out bios and their twins.
    Mirrors ``run_bios_steering.score_held_out``, generalised to the attorney/paralegal
    positive class via ``bios_pairs.score_pair``. Returns (row, requests_sent, scores_by_id)."""
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    questions = card.questions()
    test_items = ws.split("test")
    twin_items = ws.split("counterfactual")
    all_items = test_items + twin_items
    plan = ws.cache.plan([i.id for i in all_items], questions)
    sent = 0
    if plan.requests:
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
        p_attorney = confidence if result.value == POSITIVE else 1.0 - confidence
        return Verdict(item.id, result.value, p_attorney, item.reference_label,
                       item.metadata.get("gender")), p_attorney

    scores_by_id: Dict[str, float] = {}
    verdicts = []
    for i in test_items:
        v, p = verdict(i)
        verdicts.append(v)
        scores_by_id[i.id] = p
    twins = {}
    for i in twin_items:
        v, p = verdict(i)
        twins[i.metadata["counterfactual_of"]] = v
        scores_by_id[i.id] = p

    metrics = score_pair(pair=PAIR, engine=engine, verdicts=verdicts, twins=twins)
    row = metrics.as_row()
    prior = label_prior_population(ws)
    shift, corrected_accuracy = prior_corrected_accuracy(verdicts, scores_by_id, prior)
    row.update({"arm": arm, "seed": seed, "version": card.version,
               "n_labels": ws.n_labeled(SCORE_NAME),
               "label_prior_population_attorney": prior,
               "accuracy_prior_corrected_exploratory": corrected_accuracy,
               "prior_correction_shift_exploratory": shift})
    return row, sent, scores_by_id


def _mean(values: List[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def element_diagnostics(ws: Workspace, arm: str, engine: str, seed: int,
                        scores_by_id: Dict[str, float]) -> None:
    """studies/bios_attorney_elements.jsonl: for every promoted element, does it help by
    reading gender through content (a proxy: its own answer differs by gender among the true
    attorneys, on the bios as written, unrelated to the pronoun swap), or does the accuracy
    loss come from the holistic feature's fitted weight shrinking (which makes the same
    engine-confidence difference between women's and men's bios count for relatively more)?

    Written only when the current scorecard has a feature beyond the holistic one -- i.e. some
    round (this one or an earlier auto-refit) promoted something. Two kinds of row: one per
    (element, feature) with its fitted weight and its mean value by gender, as written and on
    the swapped twin, among held-out bios whose true label is "attorney"; and one
    "mechanism_separation" row per arm/seed with the top-500 four-fifths ratio recomputed three
    ways -- as fitted, with every new-element weight zeroed (isolates the holistic-reweighting
    mechanism), and with the holistic weight reset to v1's un-shrunk value (isolates the new
    element's own contribution). Neither ablation touches ``jev_flywheel/fit.py`` or the actual
    scorecard; both are computed here, read-only, from the fitted head's own weights.
    """
    card = ws.scorecard()
    score = card.score(SCORE_NAME)
    decision = score.decision
    features = list(decision.features)
    new_features = [f for f in features if f != HOLISTIC_FEATURE]
    if not new_features:
        return

    questions = card.questions()
    head = decision.head()
    test_items = ws.split("test")
    twin_items = ws.split("counterfactual")
    twin_by_orig = {i.metadata["counterfactual_of"]: i for i in twin_items}

    def feature_values(item) -> Dict[str, float]:
        answers = ws.cache.partial_answers_for(item.id, questions)
        return score.feature_vector(answers)

    true_attorneys = [i for i in test_items if i.reference_label == POSITIVE]
    elements: Dict[str, List[str]] = {}
    for f in new_features:
        elements.setdefault(f.split(".clr.")[0], []).append(f)

    rows: List[Dict[str, Any]] = []
    for elem_key, feats in elements.items():
        for feat_name in feats:
            by_gender = {"female": [], "male": []}
            by_gender_twin = {"female": [], "male": []}
            for item in true_attorneys:
                gender = item.metadata["gender"]
                fv = feature_values(item).get(feat_name)
                if fv is not None:
                    by_gender[gender].append(fv)
                twin = twin_by_orig.get(item.id)
                if twin is not None:
                    fv2 = feature_values(twin).get(feat_name)
                    if fv2 is not None:
                        by_gender_twin[gender].append(fv2)
            rows.append({
                "diagnostic": "element_gender_split", "arm": arm, "engine": engine, "seed": seed,
                "element_key": elem_key, "feature": feat_name,
                "fitted_weight_paralegal_logit": head["weights"].get("paralegal", {}).get(feat_name),
                "n_true_attorneys_female": len(by_gender["female"]),
                "n_true_attorneys_male": len(by_gender["male"]),
                "mean_feature_value_as_written_female": _mean(by_gender["female"]),
                "mean_feature_value_as_written_male": _mean(by_gender["male"]),
                "mean_feature_value_swapped_twin_female": _mean(by_gender_twin["female"]),
                "mean_feature_value_swapped_twin_male": _mean(by_gender_twin["male"]),
            })

    def ablated_head(*, zero_new: bool = False, reset_holistic: bool = False) -> Dict[str, Any]:
        ablated = dict(head)
        ablated["weights"] = {cls: dict(w) for cls, w in head["weights"].items()}
        for cls, weights in ablated["weights"].items():
            if zero_new:
                for feat_name in new_features:
                    if feat_name in weights:
                        weights[feat_name] = 0.0
            if reset_holistic and HOLISTIC_FEATURE in weights:
                weights[HOLISTIC_FEATURE] = (
                    V1_HOLISTIC_WEIGHT_PARALEGAL if cls == "paralegal" else 0.0)
        return ablated

    def rescored(ablated: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        for item in test_items + twin_items:
            fv = feature_values(item)
            _, _, detail = decide(fv, ablated)
            out[item.id] = detail["probabilities"].get(POSITIVE, 0.0)
        return out

    test_ids, twin_of = held_out_split()
    items = load_held_out_items()
    items_meta = {i: (items[i]["metadata"]["reference_label"], items[i]["metadata"]["gender"])
                 for i in test_ids}

    def ratio_at_500(scores: Dict[str, float]) -> Optional[float]:
        sc = {i: scores[i] for i in test_ids}
        twin_sc = {orig: scores[twin_of[orig]] for orig in twin_of}
        result = shortlist_metrics(items_meta, sc, twin_sc, positive_label=POSITIVE, cuts=(500,))
        return result[0]["four_fifths_ratio"]

    scores_zero = rescored(ablated_head(zero_new=True))
    scores_reset = rescored(ablated_head(reset_holistic=True))
    rows.append({
        "diagnostic": "mechanism_separation", "arm": arm, "engine": engine, "seed": seed,
        "four_fifths_ratio_top500_as_fitted": ratio_at_500(scores_by_id),
        "four_fifths_ratio_top500_new_element_zeroed": ratio_at_500(scores_zero),
        "four_fifths_ratio_top500_holistic_reset_to_v1": ratio_at_500(scores_reset),
        "note": ("as_fitted is the arm's actual result; new_element_zeroed isolates the "
                "holistic-reweighting mechanism (what the ratio would be from the refit "
                "alone, with none of the new element's own signal); "
                "holistic_reset_to_v1 isolates the new element's own contribution (what the "
                "ratio would be if the holistic feature still carried its full v1 weight, "
                "so only the new element's fitted weight differs from v1)."),
    })
    write_rows(rows, ELEMENTS_OUT)
    print(f"  element diagnostics [{arm} seed {seed}]: "
          f"as-fitted={rows[-1]['four_fifths_ratio_top500_as_fitted']}, "
          f"new-element-zeroed={rows[-1]['four_fifths_ratio_top500_new_element_zeroed']}, "
          f"holistic-reset={rows[-1]['four_fifths_ratio_top500_holistic_reset_to_v1']}")


def one_run(arm: str, seed: int, workspace_root: Path) -> Tuple[Dict[str, Any], Workspace]:
    config = ARM_CONFIG[arm]
    engine, gate = config["engine"], config["gate"]
    counter = CountingClient(_default_client()) if engine == "jev" else None
    factory = client_factory_for(engine, counter)

    ws = Workspace.init(workspace_root / "var", FIXTURES, answers=ANSWER_FILES[engine],
                        engine=engine, force=True)
    label_with_reference(ws, SCORE_NAME, N_LABELS, seed=seed)
    assert ws.n_labeled(SCORE_NAME) == N_LABELS, ws.n_labeled(SCORE_NAME)

    if engine == "jev":
        log_spend(str(date.today()), f"{arm} seed {seed}: steering round (eval top-up"
                  f"{'+ gate' if gate else ''})",
                  N_LABELS, N_LABELS * (2 if gate else 1), 0,
                  "priced as an upper bound before the round")

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

    row, serve_sent, scores_by_id = score_held_out(ws, arm, engine, seed, factory)
    write_rows([row], METRICS_OUT)
    print(f"  {arm} seed {seed}: {outcome.decision}, v{ws.version}, "
          f"accuracy {row['accuracy']} (prior-corrected, exploratory: "
          f"{row['accuracy_prior_corrected_exploratory']}), "
          f"flip {row['counterfactual_flip_rate']}")

    twin_scores_full = dict(scores_by_id)  # already keyed by item id, twin ids included
    run_shortlist(arm, engine, seed, twin_scores_full)
    element_diagnostics(ws, arm, engine, seed, twin_scores_full)

    return {"arm": arm, "seed": seed, "engine": engine, "decision": outcome.decision,
            "version": ws.version,
            "total_jev_requests": counter.requests if engine == "jev" else 0,
            "input_tokens": counter.input_tokens if counter else 0,
            "output_tokens": counter.output_tokens if counter else 0}, ws


def run_baseline(from_arm: str, seed: int, workspace_root: Path) -> None:
    """Twin averaging: no fitting. P(attorney) is the mean of ``from_arm``'s (already-fitted)
    head's probability on the bio and on its twin. Rebuilds the same workspace/labels/steering
    the source arm used (deterministic given the same seed), then simply averages at scoring
    time instead of reporting each side alone."""
    config = ARM_CONFIG[from_arm]
    engine, gate = config["engine"], config["gate"]
    counter = CountingClient(_default_client()) if engine == "jev" else None
    factory = client_factory_for(engine, counter)

    ws = Workspace.init(workspace_root / "var", FIXTURES, answers=ANSWER_FILES[engine],
                        engine=engine, force=True)
    label_with_reference(ws, SCORE_NAME, N_LABELS, seed=seed)
    run_steering(ws, SCORE_NAME, allow_spend=True, client_factory=factory,
                hitl_handler=ScriptedApprover(default=True), max_auto_requests=100000,
                invariance_max_flip_rate=GATE_RATE if gate else None)

    _, _, scores_by_id = score_held_out(ws, f"baseline-{from_arm}", engine, seed, factory)
    test_ids, twin_of = held_out_split()
    averaged = {}
    for orig in test_ids:
        twin_id = twin_of.get(orig)
        if twin_id is None:
            averaged[orig] = scores_by_id[orig]
        else:
            averaged[orig] = (scores_by_id[orig] + scores_by_id[twin_id]) / 2.0
    for orig, twin_id in twin_of.items():
        averaged[twin_id] = averaged[orig]  # symmetric by construction

    items = load_held_out_items()
    truths = {i: items[i]["metadata"]["reference_label"] for i in test_ids}
    genders = {i: items[i]["metadata"]["gender"] for i in test_ids}
    verdicts = [Verdict(i, POSITIVE if averaged[i] >= 0.5 else NEGATIVE, averaged[i],
                        truths[i], genders[i]) for i in test_ids]
    twins = {orig: Verdict(twin_id, POSITIVE if averaged[twin_id] >= 0.5 else NEGATIVE,
                           averaged[twin_id], truths[orig], genders[orig])
            for orig, twin_id in twin_of.items()}
    # Zero flips on the pronoun cue by construction: averaging makes an item and its twin
    # get exactly the same score, so this recomputes accuracy/ECE honestly and reports the
    # flip rate as a check (should print 0.0).
    metrics = score_pair(pair=PAIR, engine=engine, verdicts=verdicts, twins=twins)
    row = metrics.as_row()
    row.update({"arm": f"baseline-{from_arm}", "seed": seed, "version": None, "n_labels": N_LABELS})
    write_rows([row], METRICS_OUT)
    print(f"  baseline-{from_arm} seed {seed}: accuracy {row['accuracy']}, "
          f"flip {row['counterfactual_flip_rate']} (should be ~0)")
    run_shortlist(f"baseline-{from_arm}", engine, seed, averaged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=list(ARM_CONFIG) + ["J0", "L0"], required=False)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--scratch", type=Path, default=Path("var/bios_attorney_loop"))
    parser.add_argument("--baseline-from", choices=["L1", "J1"], default=None)
    parser.add_argument("--baseline-seed", type=int, default=1)
    parser.add_argument("--record", action="store_true", default=True)
    parser.add_argument("--no-record", dest="record", action="store_false")
    args = parser.parse_args()
    load_dotenv()

    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not SPEND_LOG.exists():
        SPEND_LOG.write_text("| when | step | items | requests priced | requests sent | notes |\n"
                             "|---|---|---:|---:|---:|---|\n")

    if args.baseline_from:
        run_baseline(args.baseline_from, args.baseline_seed,
                     args.scratch / f"baseline-{args.baseline_from}-seed{args.baseline_seed}")
        return

    if args.arm in ("J0", "L0"):
        run_engine_alone("jev" if args.arm == "J0" else "laya")
        return

    for seed in args.seeds:
        workspace_root = args.scratch / f"{args.arm}-seed{seed}"
        summary, ws = one_run(args.arm, seed, workspace_root)
        print(json.dumps(summary))
        if args.record:
            out = RECORDINGS / f"{args.arm}-seed{seed}"
            record(ws, SCORE_NAME, out, FIXTURES,
                  title=f"bios_attorney {args.arm}, seed {seed}",
                  provenance=(f"Simulated labeler (the corpus's own occupation label), 140 "
                              f"labels, one steering round "
                              f"({'with' if ARM_CONFIG[args.arm]['gate'] else 'without'} the "
                              f"gender-invariance gate), {ARM_CONFIG[args.arm]['engine']} "
                              f"engine, paralegal/attorney pair. studies/PREREGISTERED.md, "
                              f"'the learning loop on the pair that matters'"))
            print(f"  recorded to {out}")


if __name__ == "__main__":
    main()
