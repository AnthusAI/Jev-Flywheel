"""The Python side of the steering procedure.

The Tactus procedure owns the *loop*: it decides what happens next, asks the language
model for one analysis, gates everything on a human, and checkpoints. But everything
that touches numbers, files or money is deterministic Python, and it lives here, behind
four calls the procedure makes in order:

    briefing()        what the agent is shown: the scorecard, the mismatches and the human's
                      comments, and an honest account of which elements matter
    check(reply)      parse and validate the agent's proposal, apply it to the scorecard,
                      and price the Jev top-up it would need -- spending nothing
    evaluate()        fit the candidate out of fold and compare it with the incumbent; the
                      only call that can spend Jev requests, and it spends them once
    apply(...)        commit the evaluated candidate as a new scorecard version

The host keeps the candidate between calls, so ``apply`` can commit only what was
checked and evaluated. There is no way to hand it a scorecard that was not.

Two things the agent is deliberately never shown:

* **The held-out scoreboard.** It is the number that means what it says only because
  nothing optimizes against it. An agent that saw it, even indirectly, would be tuning
  to the test set one rethink at a time.
* **Its own weights as something to edit.** It sees the incumbent scorecard for
  context, but the proposal format has no field for numbers (see ``proposal.py``).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from jev_flywheel.fit import (
    Comparison, FitResult, build_training_set, compare, fit_head, latest_feedback,
    serve_summary, with_fit)
from jev_flywheel.inventory import element_inventory
from jev_flywheel.items import normalize_label
from jev_flywheel.jev import JevSession
from jev_flywheel.ladder import LadderRefusal, tier_for
from jev_flywheel.loop import describe_answer
from jev_flywheel.proposal import (
    Proposal, ProposalError, apply_proposal, diff_summary, parse_proposal)
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import predict
from jev_flywheel.workspace import Workspace

# Jev's input tokens are dominated by the item text, so a request costs about the same
# whether it carries one question or eight. Measured: 501 tokens per item for eight
# questions. Used only to give the human a rough price; the real count is reported after.
ESTIMATED_INPUT_TOKENS_PER_REQUEST = 500


def _plain(value: Any) -> Any:
    """Lua tables arrive as proxy objects; the rest of the code wants plain dicts."""
    if isinstance(value, Mapping) or (hasattr(value, "items") and hasattr(value, "keys")):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def run_sync(coro):
    """Run a coroutine to completion from code that may already be inside an event loop.

    Tactus calls host methods from within its own running loop, where ``asyncio.run``
    would raise. A fresh thread gets a fresh loop.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class FlywheelHost:
    def __init__(
        self,
        workspace: Workspace,
        score_name: str,
        *,
        allow_spend: bool = False,
        client_factory: Optional[Callable[[], Any]] = None,
        max_mismatches: int = 25,
        max_agreement_notes: int = 5,
    ):
        self.workspace = workspace
        self.score_name = score_name
        self.allow_spend = allow_spend
        self.client_factory = client_factory
        self.max_mismatches = max_mismatches
        self.max_agreement_notes = max_agreement_notes
        self._proposal: Optional[Proposal] = None
        self._candidate: Optional[Scorecard] = None
        self._fit: Optional[FitResult] = None
        self._comparison: Optional[Comparison] = None
        self._applied_version: Optional[int] = None

    # ---- what the agent is shown -----------------------------------------------

    def briefing(self) -> Dict[str, Any]:
        workspace, name = self.workspace, self.score_name
        card = workspace.scorecard()
        score = card.score(name)
        questions = card.questions()
        training = build_training_set(score, questions, workspace.cache, workspace.feedback())
        tier = tier_for(training.n_effective)
        last_fit = next((e for e in reversed(workspace.events("fit"))
                         if e.get("fitted") and e.get("score_name") == name), None)

        inventory = element_inventory(
            score, training.rows, training.labels, training.weights) if training.n else []
        mismatches, agreements = self._feedback_examples(card, score, questions)
        return {
            "score": name,
            "scorecard_yaml": card.to_yaml(),
            "summary": {
                "scorecard_version": workspace.version,
                "n_labeled": training.n,
                "n_effective": round(training.n_effective, 1),
                "capability_tier": tier.name,
                "feature_budget": tier.feature_budget(training.n_effective),
                "features_in_use": len(score.decision.features) if score.decision else 0,
                "classes": list(score.decision.classes) if score.decision else [],
                "label_distribution": _count(training.labels),
                "disagreements": sum(1 for m in mismatches),
                # Out-of-fold numbers from the last fit. The held-out scoreboard is
                # withheld on purpose: see the module docstring.
                "last_fit": None if last_fit is None else {
                    "out_of_fold_accuracy": last_fit.get("oof_accuracy"),
                    "out_of_fold_brier": last_fit.get("oof_brier"),
                },
            },
            "mismatches": mismatches,
            "commented_agreements": agreements,
            "element_inventory": inventory,
        }

    def _feedback_examples(self, card: Scorecard, score, questions):
        latest = [f for f in latest_feedback(self.workspace.feedback(), self.score_name).values()
                  if f.label is not None]
        rows = []
        for record in latest:
            answers = self.workspace.cache.partial_answers_for(record.item_id, questions)
            result = predict(score, answers)
            wrong = normalize_label(record.initial_answer_value) != normalize_label(
                record.final_answer_value)
            rows.append({
                "wrong": wrong, "commented": bool(record.edit_comment_value),
                "shown_confidence": record.metadata.get("shown_confidence") or 0.0,
                "entry": {
                    "item_id": record.item_id,
                    "text": self.workspace.item(record.item_id).text,
                    "we_said": record.initial_answer_value,
                    "confidence_when_shown": _round(record.metadata.get("shown_confidence")),
                    "human_said": record.final_answer_value,
                    "human_comment": record.edit_comment_value,
                    "element_answers": {
                        wire: describe_answer(answer)
                        for wire, answer in answers.items() if wire != score.question_name},
                    "top_drivers_now": [
                        f"{d['feature']} ({d['contribution']:+.2f})"
                        for d in (result.metadata.get("decision") or {}).get("top_contributions", [])],
                }})
        wrong = sorted((r for r in rows if r["wrong"]),
                       key=lambda r: (not r["commented"], -r["shown_confidence"]))
        notes = [r for r in rows if not r["wrong"] and r["commented"]]
        return ([r["entry"] for r in wrong[:self.max_mismatches]],
                [r["entry"] for r in notes[:self.max_agreement_notes]])

    # ---- validating a proposal, spending nothing --------------------------------

    def check(self, reply: Any) -> Dict[str, Any]:
        """Parse the agent's proposal, apply it, and price what evaluating it would cost."""
        self._proposal = self._candidate = self._fit = self._comparison = None
        problems: List[str] = []
        try:
            proposal = parse_proposal(_plain(reply) if not isinstance(reply, str) else reply)
            card = self.workspace.scorecard()
            candidate = apply_proposal(card, self.score_name, proposal)
        except ProposalError as error:
            return {"ok": False, "noop": False, "problems": [str(error)], "plan": None}

        score = candidate.score(self.score_name)
        diff = diff_summary(card, candidate, self.score_name)
        noop = proposal.is_noop
        training = build_training_set(
            score, candidate.questions(), self.workspace.cache, self.workspace.feedback())
        # Items lacking answers will be topped up, so they count toward the budget: it is
        # what the fit will see once the answers arrive.
        eligible = training.n_effective + len(training.needs_answers)
        tier = tier_for(eligible)
        budget = tier.feature_budget(eligible)
        if tier.name != "hold" and len(score.decision.features) > budget:
            problems.append(
                f"{len(score.decision.features)} features exceeds the budget of {budget} that "
                f"about {eligible:.0f} labels supports. Retire an element, or add fewer.")
        if problems:
            return {"ok": False, "noop": noop, "problems": problems, "plan": None, "diff": diff}

        plan = self.workspace.cache.plan(
            [f.item_id for f in latest_feedback(
                self.workspace.feedback(), self.score_name).values() if f.label is not None],
            candidate.questions())
        self._proposal, self._candidate = proposal, candidate
        tokens = plan.requests * ESTIMATED_INPUT_TOKENS_PER_REQUEST
        return {
            "ok": True, "noop": noop, "problems": [], "diff": diff,
            "root_cause": proposal.root_cause,
            "n_features": len(score.decision.features), "feature_budget": budget,
            "plan": {
                "requests": plan.requests, "missing_answers": plan.missing_answers,
                "estimated_input_tokens": tokens,
            },
            "summary_text": _check_text(diff, len(score.decision.features), budget,
                                        plan.requests, tokens),
        }

    # ---- evaluating it -----------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Fit the checked candidate out of fold and compare it with the incumbent.

        This is the only call that can spend Jev requests, and it spends them once:
        one request per labeled item that lacks an answer, carrying only the missing
        questions.
        """
        if self._candidate is None:
            return {"status": "error", "reason": "no valid proposal has been checked"}
        candidate, name = self._candidate, self.score_name
        score, questions = candidate.score(name), candidate.questions()

        training = build_training_set(
            score, questions, self.workspace.cache, self.workspace.feedback())
        if training.needs_answers:
            if not self.allow_spend:
                return {"status": "needs_spend", "requests": len(training.needs_answers),
                        "reason": "the candidate needs answers that are not cached, and "
                                  "this run is not allowed to call Jev"}
            self._top_up(training.needs_answers, questions)
            training = build_training_set(
                score, questions, self.workspace.cache, self.workspace.feedback())

        try:
            result = fit_head(training, score)
        except LadderRefusal as refusal:
            return {"status": "refused", "reason": str(refusal), "promote": False}
        if not result.fitted:
            return {"status": "held", "reason": result.reason, "promote": False}

        incumbent_card = self.workspace.scorecard()
        incumbent = serve_summary(
            incumbent_card.score(name), incumbent_card.questions(), self.workspace.cache, training)
        comparison = compare(result, incumbent)
        self._fit, self._comparison = result, comparison
        return {
            "status": "fitted", "promote": comparison.promote, "reasons": comparison.reasons,
            "tier": result.tier.name, "n_train": result.n, "n_effective": round(result.n_effective, 1),
            "candidate": _metrics(comparison.candidate),
            "incumbent": _metrics(comparison.incumbent),
            "summary_text": _evaluation_text(result, comparison),
        }

    def _top_up(self, item_ids: List[str], questions: Mapping[str, Any]) -> None:
        wanted = set(item_ids)
        items = [i for i in self.workspace.items if i.id in wanted]
        session = JevSession(client_factory=self.client_factory) if self.client_factory \
            else JevSession()
        run_sync(self.workspace.cache.fill(session, items, questions))

    # ---- committing it -----------------------------------------------------------

    def apply(self, provenance: Any = None) -> Dict[str, Any]:
        """Commit the evaluated candidate. Refuses anything that was not evaluated and better."""
        if self._applied_version is not None:
            return {"version": self._applied_version, "already_applied": True}
        if self._fit is None or self._comparison is None or not self._comparison.promote:
            raise RuntimeError("only a candidate that was evaluated and beat the incumbent "
                               "can be applied")
        details = _plain(provenance) or {}
        card = with_fit(self._candidate, self.score_name, self._fit)
        version = self.workspace.commit_scorecard(card, kind="steer", provenance={
            **details, "proposal": self._proposal.summary() if self._proposal else {},
            "root_cause": self._proposal.root_cause if self._proposal else "",
            "fit": self._fit.provenance,
        })
        self._applied_version = version
        return {"version": version, "already_applied": False}


def _check_text(diff: Mapping[str, Any], n_features: int, budget: int, requests: int,
                tokens: int) -> str:
    """The proposed change, in words, for the human deciding whether to approve it."""
    changes = []
    if diff["added"]:
        changes.append("add " + ", ".join(diff["added"]))
    if diff["retired"]:
        changes.append("retire " + ", ".join(diff["retired"]))
    if diff["reworded"]:
        changes.append("reword " + ", ".join(diff["reworded"]))
    if diff["holistic_reworded"]:
        changes.append("reword the holistic question")
    features = [f"+{f}" for f in diff["features_added"]] + [f"-{f}" for f in diff["features_removed"]]
    cost = (f"{requests} Jev requests (about {tokens:,} input tokens) to evaluate"
            if requests else "no new Jev requests: every answer is already cached")
    return (f"Changes: {'; '.join(changes) or 'none'}\n"
            f"Features: {' '.join(features) or 'unchanged'} ({n_features} of a budget of {budget})\n"
            f"Cost: {cost}")


def _evaluation_text(result: FitResult, comparison: Comparison) -> str:
    """How the candidate did against the incumbent, out of fold, in words."""
    def line(label: str, summary) -> str:
        return (f"  {label}: accuracy {summary.accuracy:.3f}, ECE {summary.ece:.3f}, "
                f"Brier {summary.brier:.3f}")
    text = (f"Out of fold on {result.n} labels ({result.n_effective:.0f} effective, "
            f"tier {result.tier.name}):\n{line('candidate', comparison.candidate)}\n"
            f"{line('incumbent', comparison.incumbent)}")
    if comparison.reasons:
        text += "\nNot promoted: " + "; ".join(comparison.reasons)
    return text


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)


def _metrics(summary) -> Dict[str, Any]:
    return {"accuracy": round(summary.accuracy, 4), "ece": round(summary.ece, 4),
            "brier": round(summary.brier, 4), "n": summary.n}


def _count(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def describe(host_result: Mapping[str, Any]) -> str:
    """A one-line rendering of an evaluation, for logs."""
    return json.dumps(dict(host_result), default=str)[:400]
