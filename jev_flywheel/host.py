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
        max_labeled_sample: int = 40,
    ):
        self.workspace = workspace
        self.score_name = score_name
        self.allow_spend = allow_spend
        self.client_factory = client_factory
        self.max_mismatches = max_mismatches
        self.max_agreement_notes = max_agreement_notes
        self.max_labeled_sample = max_labeled_sample
        self._proposal: Optional[Proposal] = None
        self._candidate: Optional[Scorecard] = None
        self._fit: Optional[FitResult] = None
        self._comparison: Optional[Comparison] = None
        self._applied_version: Optional[int] = None
        # The analyst's raw reply, kept so a round can be recorded and replayed exactly.
        self.last_reply: Optional[str] = None
        self.last_discovery_reply: Optional[str] = None

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
            "labeled_sample": self._labeled_sample(),
            "blind_sample": self.blind_sample(),
            "element_inventory": inventory,
        }

    def blind_sample(self, per_group: int = 40) -> Dict[str, Any]:
        """Labeled texts with the task stripped out: Group A and Group B, nothing else.

        The analyst that sees the scorecard is frame-locked by it -- told it is improving a
        sentiment score, it proposes sentiment features, which is how a factor orthogonal to
        sentiment stays invisible. This view removes the frame: no score name, no criteria, no
        elements, no mention of what the groups mean. It is a pure induction task, which is
        what "find the rule these labels follow" actually is.
        """
        records = [f for f in latest_feedback(self.workspace.feedback(), self.score_name).values()
                   if f.label is not None]
        groups: Dict[str, List[Any]] = {}
        for record in records:
            groups.setdefault(record.final_answer_value, []).append(record)
        names = {label: f"Group {chr(65 + i)}" for i, label in enumerate(sorted(groups))}
        out: Dict[str, List[str]] = {name: [] for name in names.values()}
        for label, members in groups.items():
            for record in members[-per_group:]:
                out[names[label]].append(self.workspace.item(record.item_id).text)
        return out

    def _remember_discovery(self, reply: Any) -> None:
        if reply:
            self.last_discovery_reply = reply if isinstance(reply, str) else json.dumps(_plain(reply))

    def check_combined(self, *replies: Any) -> Dict[str, Any]:
        """Merge several proposals into one, then check it.

        Candidates from the blind view and from the error analysis are pooled rather than
        filtered by either. Screening them is nearly free -- every question rides in the same
        request, so N candidates cost the same number of Jev requests as one -- and the fit,
        not a frame-locked model, decides which survive.

        The first reply is the analyst's and is required: if it will not parse, that is a
        checkable failure the caller can ask it to repair. Later replies are the blind pass,
        which is best-effort -- losing it costs some candidates, not the round.
        """
        if replies and replies[0]:
            first = replies[0]
            try:
                parse_proposal(_plain(first) if not isinstance(first, str) else first)
            except ProposalError as error:
                self.last_reply = first if isinstance(first, str) else json.dumps(_plain(first))
                return {"ok": False, "noop": False, "problems": [str(error)], "plan": None}
        if len(replies) > 1:
            self._remember_discovery(replies[1])
        merged: Dict[str, Any] = {"root_cause": "", "add_elements": [],
                                  "retire_elements": [], "reword_elements": []}
        seen = set()
        for reply in replies:
            if not reply:
                continue
            try:
                proposal = parse_proposal(_plain(reply) if not isinstance(reply, str) else reply)
            except ProposalError:
                continue                       # one unusable reply must not sink the round
            for added in proposal.add:
                if added.key not in seen:
                    seen.add(added.key)
                    merged["add_elements"].append({
                        "key": added.key, "question_type": added.question_type,
                        "instructions": added.instructions, "criteria": added.criteria})
            merged["retire_elements"].extend(proposal.retire)
            merged["reword_elements"].extend(
                [{"key": r.key, "instructions": r.instructions} for r in proposal.reword])
            if proposal.root_cause and not merged["root_cause"]:
                merged["root_cause"] = proposal.root_cause
        return self.check(merged)

    def _labeled_sample(self) -> List[Dict[str, Any]]:
        """A sample of labeled items and their labels, right or wrong, newest first.

        The mismatch list answers "what are we getting wrong". It cannot answer "what
        decides the label", because a regularity the scorecard already exploits produces no
        errors to look at: on the corpus we ship, items about sport skew positive, and that
        shows up in the mismatches as almost nothing, since those items are mostly already
        right. An analyst shown only failures has a structural blind spot for base rates.
        So it also gets a plain, balanced sample of the labeled data.
        """
        records = [f for f in latest_feedback(self.workspace.feedback(), self.score_name).values()
                   if f.label is not None]
        by_label: Dict[str, List[Any]] = {}
        for record in records:
            by_label.setdefault(record.final_answer_value, []).append(record)
        per = max(self.max_labeled_sample // max(len(by_label), 1), 1)
        picked = [r for group in by_label.values() for r in group[-per:]]
        return [{"text": self.workspace.item(r.item_id).text, "human_label": r.final_answer_value}
                for r in picked]

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
        self.last_reply = reply if isinstance(reply, str) else json.dumps(_plain(reply))
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
        # Evaluating needs answers only for the labeled items. *Serving* the candidate, to
        # choose the next question or to score the held-out split, needs them for every
        # item, and a reworded question makes every stored answer to it stale. The human
        # should know both prices before approving.
        serving = self.workspace.cache.plan(
            [i.id for i in self.workspace.items], candidate.questions())
        self._proposal, self._candidate = proposal, candidate
        tokens = plan.requests * ESTIMATED_INPUT_TOKENS_PER_REQUEST
        return {
            "ok": True, "noop": noop, "problems": [], "diff": diff,
            "root_cause": proposal.root_cause,
            "n_features": len(score.decision.features), "feature_budget": budget,
            "plan": {
                "requests": plan.requests, "missing_answers": plan.missing_answers,
                "estimated_input_tokens": tokens,
                "serving_requests": serving.requests,
            },
            "summary_text": _check_text(diff, len(score.decision.features), budget,
                                        plan.requests, tokens, serving.requests),
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
            report = self._top_up(training.needs_answers, questions)
            training = build_training_set(
                score, questions, self.workspace.cache, self.workspace.feedback())
            # Fitting on whatever survived would quietly train on a fraction of the labels
            # and report metrics as if it had them all. Say the top-up failed instead.
            still_missing = len(training.needs_answers)
            if still_missing > max(1, 0.2 * (training.n + still_missing)):
                return {
                    "status": "top_up_failed", "promote": False,
                    "requested": report.requested, "failed": report.failures,
                    "still_missing": still_missing,
                    "reason": f"asking Jev for the new answers failed for {still_missing} of "
                              f"{training.n + still_missing} labeled items"
                              + (f" ({report.errors[0]})" if report.errors else "")
                              + ". Nothing was fit. Check the API key and try again; answers "
                                "already fetched are kept."}

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

    def _top_up(self, item_ids: List[str], questions: Mapping[str, Any]):
        wanted = set(item_ids)
        items = [i for i in self.workspace.items if i.id in wanted]
        session = JevSession(client_factory=self.client_factory) if self.client_factory \
            else JevSession()
        return run_sync(self.workspace.cache.fill(session, items, questions))

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
            "analyst_reply": self.last_reply,
            "fit": self._fit.provenance,
        })
        self._applied_version = version
        return {"version": version, "already_applied": False}


def _check_text(diff: Mapping[str, Any], n_features: int, budget: int, requests: int,
                tokens: int, serving_requests: int = 0) -> str:
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
    text = (f"Changes: {'; '.join(changes) or 'none'}\n"
            f"Features: {' '.join(features) or 'unchanged'} ({n_features} of a budget of {budget})\n"
            f"Cost: {cost}")
    if serving_requests > requests:
        text += (f"\nServing it on every item (to pick questions and score the held-out "
                 f"split) would need {serving_requests:,} requests in all"
                 + ("; rewording the holistic question makes every stored answer to it stale"
                    if diff.get("holistic_reworded") else ""))
    return text


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
