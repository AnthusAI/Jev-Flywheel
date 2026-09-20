"""The flywheel's operations, in one place.

The console, the command line and the Tactus procedure's host module all call into
these functions and hold no logic of their own. That is deliberate: there is one
definition of "ask the next question", "record a judgement" and "refit and maybe
promote", and it is spec'd here, not re-implemented three times with three subtly
different rules.

The cycle, end to end:

    next_question   pick the item a label would teach most, record the probability
                    it was picked with
    record_label    the human agrees, disagrees (with the correct label) or skips,
                    with an optional comment; written as a FeedbackItem
    refit           re-estimate the head from all labels; promote it only if it beats
                    the incumbent out of fold
    status          which triggers have fired, and how far off the others are
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jev_flywheel.fit import (
    FitResult, build_training_set, compare, fit_head, serve_summary, with_fit)
from jev_flywheel.items import (
    LABEL_SOURCE_FINAL, LABEL_SOURCE_UNRESOLVED, FeedbackItem, Item, normalize_label, now)
from jev_flywheel.ladder import CAPABILITY_LADDER_V1, LadderRefusal, distance_to_next, tier_for
from jev_flywheel.scoring import ScoreResult
from jev_flywheel.selection import Selection, SelectionPolicy, build_candidates, choose
from jev_flywheel.steering import SteeringPolicy, Trigger, evaluate
from jev_flywheel.workspace import Workspace

AGREE, DISAGREE, SKIP = "agree", "disagree", "skip"


def describe_answer(answer: Dict[str, Any]) -> str:
    """One short line for a Jev answer, for the console."""
    if "noul" in answer:
        return f"{float(answer['noul']):.0%} yes"
    if "choice" in answer:
        probabilities = answer.get("probabilities") or {}
        top = probabilities.get(answer["choice"])
        return f"{answer['choice']}" + (f" ({top:.0%})" if top is not None else "")
    if "score" in answer:
        legend = answer.get("legend") or {}
        level = legend.get(str(int(round(float(answer["score"])))))
        return f"{float(answer['score']):.1f}" + (f" ({level})" if level else "")
    return "?"


@dataclass
class Question:
    """Everything the human is shown about one item."""

    item: Item
    result: ScoreResult
    selection: Selection
    scorecard_version: int
    answers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)


def next_question(
    workspace: Workspace, score_name: str, rng: random.Random,
    policy: SelectionPolicy = SelectionPolicy(), *, pool_split: str = "pool",
) -> Optional[Question]:
    """The most informative unlabeled item in the pool, or None if it is exhausted.

    Only the pool is ever offered. The test split is the scoreboard: no human sees it
    and no selection rule touches it, so accuracy measured there means what it says.
    """
    card = workspace.scorecard()
    score = card.score(score_name)
    questions = card.questions()
    candidates = build_candidates(
        score, questions, workspace.cache, workspace.split(pool_split),
        workspace.labeled_ids(score_name), policy=policy)
    if not candidates:
        return None
    selection = choose(candidates, rng, policy)
    item_id = selection.candidate.item_id
    return Question(
        item=workspace.item(item_id), result=selection.candidate.result, selection=selection,
        scorecard_version=workspace.version,
        answers=workspace.cache.partial_answers_for(item_id, questions),
        classes=list(score.decision.classes) if score.decision else [])


def record_label(
    workspace: Workspace, question: Question, verdict: str, *,
    correct_label: Optional[str] = None, comment: Optional[str] = None,
    editor: str = "human",
) -> FeedbackItem:
    """Write the human's judgement as a FeedbackItem.

    ``initial_answer_value`` is what we predicted and ``final_answer_value`` is what
    the human settled on; they are equal when the human agreed. The selection record
    (its propensity above all) rides in the metadata so the fit can weight this label.
    """
    score_name = question.result.score_name
    predicted = question.result.value
    metadata = {
        **question.selection.record(),
        "scorecard_version": question.scorecard_version,
        "raw_confidence": question.result.raw_confidence,
        "shown_confidence": question.result.confidence,
    }
    common = dict(
        id=f"fb-{question.item.id}-{int(time.time() * 1000)}",
        item_id=question.item.id, score_name=score_name, initial_answer_value=predicted,
        edit_comment_value=comment or None, editor_name=editor, edited_at=now(),
        cache_key=f"{question.item.id}:{score_name}:v{question.scorecard_version}")

    if verdict == SKIP:
        # Recorded so selection does not immediately offer the item again, but with
        # no usable label: an unresolved label source is never trained on.
        return _store(workspace, FeedbackItem(
            **common, final_answer_value=None, is_agreement=None,
            label_source=LABEL_SOURCE_UNRESOLVED, metadata={**metadata, "skipped": True}))
    if verdict == AGREE:
        final, agreement = predicted, True
    elif verdict == DISAGREE:
        if not correct_label:
            raise ValueError("disagreeing needs the correct label")
        allowed = {normalize_label(c): c for c in question.classes}
        if normalize_label(correct_label) not in allowed:
            raise ValueError(f"{correct_label!r} is not one of {question.classes}")
        final = allowed[normalize_label(correct_label)]
        agreement = normalize_label(final) == normalize_label(predicted)
    else:
        raise ValueError(f"unknown verdict {verdict!r}; expected agree, disagree or skip")
    return _store(workspace, FeedbackItem(
        **common, final_answer_value=final, is_agreement=agreement,
        label_source=LABEL_SOURCE_FINAL, metadata=metadata))


def _store(workspace: Workspace, record: FeedbackItem) -> FeedbackItem:
    workspace.add_feedback(record)
    return record


@dataclass
class RefitOutcome:
    """What a refit attempt did, and why."""

    status: str                          # promoted, rejected, held, refused
    reasons: List[str] = field(default_factory=list)
    result: Optional[FitResult] = None
    version: Optional[int] = None
    needs_answers: int = 0
    comparison: Any = None

    @property
    def promoted(self) -> bool:
        return self.status == "promoted"


def refit(workspace: Workspace, score_name: str, *, seed: int = 0,
          dry_run: bool = False) -> RefitOutcome:
    """Re-estimate the head from every label so far; promote it only if it earns it.

    The candidate's numbers are out-of-fold and the incumbent's are its own served
    predictions on the same labeled items and weights, so the comparison is honest
    and conservative. A candidate that does not win is recorded and dropped; the
    incumbent stays.
    """
    card = workspace.scorecard()
    score = card.score(score_name)
    questions = card.questions()
    training = build_training_set(
        score, questions, workspace.cache, workspace.feedback())

    def log(outcome: RefitOutcome) -> RefitOutcome:
        result = outcome.result
        workspace.log_event(
            "fit", score_name, fitted=bool(result and result.fitted), status=outcome.status,
            promoted=outcome.promoted, reasons=outcome.reasons,
            tier=result.tier.name if result else tier_for(training.n_effective).name,
            n_effective=round(training.n_effective, 2), needs_answers=len(training.needs_answers),
            log_loss=result.log_loss if result and result.fitted else None,
            oof_accuracy=result.metrics.accuracy if result and result.fitted else None,
            oof_brier=result.metrics.brier if result and result.fitted else None,
            fit_id=result.provenance.get("fit_id") if result and result.fitted else None,
            new_version=outcome.version)
        return outcome

    try:
        result = fit_head(training, score, seed=seed)
    except LadderRefusal as refusal:
        return log(RefitOutcome("refused", [str(refusal)], needs_answers=len(training.needs_answers)))
    if not result.fitted:
        return log(RefitOutcome("held", [result.reason or "too few labels"], result,
                                needs_answers=len(training.needs_answers)))

    incumbent = serve_summary(score, questions, workspace.cache, training)
    comparison = compare(result, incumbent)
    if not comparison.promote:
        return log(RefitOutcome("rejected", comparison.reasons, result,
                                needs_answers=len(training.needs_answers), comparison=comparison))
    version = None
    if not dry_run:
        version = workspace.commit_scorecard(
            with_fit(card, score_name, result), kind="fit", provenance=result.provenance)
    return log(RefitOutcome("promoted", [], result, version,
                            len(training.needs_answers), comparison))


@dataclass
class Status:
    score_name: str
    version: int
    n_labeled: int
    n_effective: float
    tier: str
    next_tier: Optional[str]
    distance_to_next: Optional[float]
    triggers: Dict[str, Trigger]


def status(workspace: Workspace, score_name: str,
           policy: SteeringPolicy = SteeringPolicy()) -> Status:
    """Where the flywheel stands, and what it thinks is worth doing next."""
    card = workspace.scorecard()
    score = card.score(score_name)
    training = build_training_set(
        score, card.questions(), workspace.cache, workspace.feedback()) \
        if score.decision else None
    n_effective = training.n_effective if training else 0.0
    state = workspace.steering_state(score_name, n_effective)
    ladder = CAPABILITY_LADDER_V1
    tier = tier_for(n_effective, ladder)
    upcoming = distance_to_next(n_effective, ladder)
    following = None
    if upcoming is not None:
        following = ladder[ladder.index(tier) + 1].name
    return Status(score_name, workspace.version, state.n_labeled, n_effective, tier.name,
                  following, upcoming, evaluate(state, policy))
