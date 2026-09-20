"""Fitting the decision head from human feedback.

scikit-learn and numpy are used here and nowhere on the serving path: the fitted
result is a plain dict of named weights that ``head.py`` evaluates with ``math``
alone, which is what lets a fitted head live in a YAML file, be read by a person,
and be diffed between versions.

The order of operations matters, and each step exists because skipping it fails
silently:

1. **Only trusted labels.** A feedback item that falls back to the AI's own earlier
   prediction teaches the head to imitate the incumbent, and it looks like it is
   working because it agrees with the baseline.
2. **Inverse-propensity weights.** Active selection makes the labeled set a biased
   sample; without the correction the head is calibrated to a world that does not
   exist (``sampling.py``).
3. **Full feature coverage, or refuse.** Serving may impute zero for a missing
   feature; training may not. Imputed rows bias a newly proposed element's weight
   toward zero, and then the optimizer concludes its own proposal was useless and
   retires it.
4. **Effective sample size picks the tier** (``ladder.py``), and asking for more
   than the tier permits is refused with a message saying what would suffice.
5. **Cross-validate, then calibrate on the out-of-fold predictions** -- never on
   in-sample ones (``calibrate.py``).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from jev_flywheel.answers import AnswerCache
from jev_flywheel.calibrate import OutOfFoldPredictions, fit_calibration
from jev_flywheel.evaluate import Summary, summarize
from jev_flywheel.items import FeedbackItem, agrees, normalize_label
from jev_flywheel.jev import fingerprint
from jev_flywheel.ladder import (
    CAPABILITY_LADDER_V1, LADDER_NAME, LadderRefusal, Tier, distance_to_next, tier_for)
from jev_flywheel.sampling import inverse_propensity_weights, kish_n_effective
from jev_flywheel.scorecard import Score, Scorecard
from jev_flywheel.scoring import predict


@dataclass
class TrainingSet:
    """Labeled items with their features, weights and provenance."""

    item_ids: List[str] = field(default_factory=list)
    rows: List[Dict[str, float]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    cells: List[Optional[str]] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)
    needs_answers: List[str] = field(default_factory=list)
    question_set_fingerprint: str = ""

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def n_effective(self) -> float:
        return kish_n_effective(self.weights)


def latest_feedback(feedback: Iterable[FeedbackItem], score_name: str) -> Dict[str, FeedbackItem]:
    """The most recent feedback per item. A revisit supersedes; it never duplicates."""
    latest: Dict[str, FeedbackItem] = {}
    for record in feedback:
        if record.score_name == score_name:
            latest[record.item_id] = record
    return latest


def build_training_set(
    score: Score,
    questions: Mapping[str, Mapping[str, Any]],
    cache: AnswerCache,
    feedback: Iterable[FeedbackItem],
    *,
    default_propensity: Optional[float] = None,
    max_ratio: Optional[float] = 20.0,
) -> TrainingSet:
    """Assemble what a fit needs, counting everything it had to leave out.

    ``needs_answers`` lists items that have a usable label but no complete set of
    cached answers -- typically because a new element was just proposed. They are
    left out of the fit and reported, so the caller can top up the cache and try
    again, rather than have the fit quietly train on fewer items than it appears to.
    """
    if score.decision is None:
        raise ValueError(f"score {score.name!r} has no decision block, so there is nothing to fit")
    canonical = {normalize_label(c): c for c in score.decision.classes}
    training = TrainingSet(question_set_fingerprint=fingerprint(dict(questions)))
    propensities: List[float] = []

    def drop(reason: str) -> None:
        training.dropped[reason] = training.dropped.get(reason, 0) + 1

    for item_id, record in latest_feedback(feedback, score.name).items():
        label = record.label
        if label is None:
            drop("no_trusted_label")
            continue
        cls = canonical.get(normalize_label(label))
        if cls is None:
            drop("label_not_in_classes")
            continue
        propensity = record.propensity if record.propensity is not None else default_propensity
        if propensity is None:
            drop("no_propensity")
            continue
        answers = cache.answers_for(item_id, questions)
        if answers is None:
            training.needs_answers.append(item_id)
            continue
        training.item_ids.append(item_id)
        training.rows.append(score.feature_vector(answers))
        training.labels.append(cls)
        training.cells.append(record.confusion_cell)
        propensities.append(propensity)

    training.weights = inverse_propensity_weights(propensities, max_ratio=max_ratio)
    return training


def build_matrix(rows: Sequence[Mapping[str, float]], feature_names: Sequence[str]):
    """The feature matrix in declared order. Refuses any row with a missing feature."""
    import numpy as np

    matrix = []
    for index, row in enumerate(rows):
        missing = [name for name in feature_names if name not in row]
        if missing:
            raise ValueError(
                f"row {index} is missing features {missing}; training needs full coverage, "
                "because imputing a neutral value biases a new feature's weight toward zero")
        matrix.append([row[name] for name in feature_names])
    return np.array(matrix, dtype=float)


@dataclass
class FitResult:
    """Everything a fit produced, including the decision not to fit."""

    status: str                       # "fitted" or "held"
    tier: Tier
    n: int
    n_effective: float
    reason: Optional[str] = None
    head: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    oof: Optional[OutOfFoldPredictions] = None
    metrics: Optional[Summary] = None
    log_loss: Optional[float] = None
    chosen_c: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.status == "fitted"


def _weighted_log_loss(proba, labels_index, weights) -> float:
    import numpy as np

    picked = proba[np.arange(len(labels_index)), labels_index]
    return float(-(weights * np.log(np.clip(picked, 1e-9, 1.0))).sum() / weights.sum())


def fit_head(
    training: TrainingSet,
    score: Score,
    *,
    ladder: Tuple[Tier, ...] = CAPABILITY_LADDER_V1,
    folds: int = 5,
    seed: int = 0,
) -> FitResult:
    """Fit the score's decision head, if the evidence supports it.

    Returns a ``held`` result rather than raising when the sample is too small to
    fit anything: not being able to fit yet is an ordinary state for a young
    scorecard, not an error. Raises ``LadderRefusal`` when a fit was possible but
    what the score asks for -- too many features, a class with no labels -- is not.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    decision = score.decision
    if decision is None:
        raise ValueError(f"score {score.name!r} has no decision block")
    n, n_eff = training.n, training.n_effective
    tier = tier_for(n_eff, ladder)
    base = {"ladder": LADDER_NAME, "tier": tier.name, "n_train": n,
            "n_effective": round(n_eff, 2)}

    if tier.name == "hold":
        gap = distance_to_next(n_eff, ladder)
        return FitResult(
            "held", tier, n, n_eff, provenance=base,
            reason=f"{n_eff:.1f} effective labels is below the {ladder[1].min_n_effective:.0f} "
                   f"needed to fit; {gap:.1f} more are needed. Keeping the incumbent.")

    features = list(decision.features)
    budget = tier.feature_budget(n_eff)
    if len(features) > budget:
        upcoming = distance_to_next(n_eff, ladder)
        raise LadderRefusal(
            f"{len(features)} features but {n_eff:.0f} effective labels supports {budget} at "
            f"tier {tier.name!r}. Drop features, or collect "
            f"{'more labels' if upcoming is None else f'{upcoming:.0f} more effective labels for the next tier'}.")

    counts = Counter(training.labels)
    if len(counts) < 2:
        raise LadderRefusal(
            f"every label is {next(iter(counts))!r}; a fit needs both classes represented")
    if min(counts.values()) < 3:
        rare = min(counts, key=counts.get)
        raise LadderRefusal(
            f"only {counts[rare]} label(s) for {rare!r}; cross-validation needs at least 3 per class")

    X = build_matrix(training.rows, features)
    classes = sorted(counts)
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[label] for label in training.labels])
    w = np.array(training.weights, dtype=float)
    splitter = StratifiedKFold(n_splits=min(folds, min(counts.values())), shuffle=True,
                               random_state=seed)

    def oof_proba(c: float):
        out = np.zeros((n, len(classes)))
        for train, test in splitter.split(X, y):
            model = LogisticRegression(C=c, max_iter=2000)
            model.fit(X[train], y[train], sample_weight=w[train])
            out[test] = model.predict_proba(X[test])
        return out

    scored = {c: oof_proba(c) for c in tier.c_grid}
    best_c = min(scored, key=lambda c: _weighted_log_loss(scored[c], y, w))
    proba = scored[best_c]

    predicted = proba.argmax(axis=1)
    confidences = proba.max(axis=1).tolist()
    correct = (predicted == y).astype(int).tolist()
    oof = OutOfFoldPredictions(confidences, correct, w.tolist())
    calibration = fit_calibration(oof, tier.calibration)

    final = LogisticRegression(C=best_c, max_iter=2000)
    final.fit(X, y, sample_weight=w)
    head = _serving_head(final, classes, features, list(decision.classes))

    population = {c: float(w[y == index[c]].sum() / w.sum()) for c in classes}
    fit_id = "dh-" + hashlib.sha256(json.dumps(
        [sorted(zip(training.item_ids, training.labels)), features, best_c],
        sort_keys=True).encode()).hexdigest()[:10]
    provenance = {
        **base, "fit_id": fit_id, "model": head["model"], "regularization_c": best_c,
        "features": features, "feature_budget": budget,
        "question_set_fingerprint": training.question_set_fingerprint,
        "cv": {"folds": splitter.get_n_splits(), "seed": seed, "selected_on": "weighted_log_loss"},
        "label_prior_population": population,
        "cell_census": dict(Counter(c for c in training.cells if c)),
        "calibration_method": calibration["method"],
    }
    if training.dropped:
        provenance["dropped"] = dict(training.dropped)
    return FitResult(
        "fitted", tier, n, n_eff, head=head, calibration=calibration, oof=oof,
        metrics=summarize(confidences, correct, w.tolist()),
        log_loss=_weighted_log_loss(proba, y, w), chosen_c=best_c, provenance=provenance)


def _serving_head(model, sk_classes: List[str], features: List[str],
                  declared: List[str]) -> Dict[str, Any]:
    """A fitted sklearn model in the plain-dict form ``head.py`` serves from."""
    if len(sk_classes) == 2:
        # sklearn stores one coefficient vector, for the second class; the first is
        # the reference and needs no weights of its own.
        weights = {sk_classes[1]: {
            "intercept": float(model.intercept_[0]),
            **{name: float(value) for name, value in zip(features, model.coef_[0])}}}
    else:
        weights = {cls: {"intercept": float(b),
                         **{name: float(value) for name, value in zip(features, coef)}}
                   for cls, b, coef in zip(sk_classes, model.intercept_, model.coef_)}
    ordered = [c for c in declared if c in sk_classes] + [c for c in sk_classes if c not in declared]
    return {"model": "multinomial_logistic", "classes": ordered, "weights": weights}


def with_fit(scorecard: Scorecard, score_name: str, result: FitResult) -> Scorecard:
    """A copy of the scorecard with the fitted head, calibration and provenance in place.

    Only ``parameters.weights``, ``calibration`` and ``provenance`` change. The
    elements, instructions and criteria -- the part a person or the steering agent
    edits -- are untouched, which is what keeps numbers and words separately owned.
    """
    if not result.fitted:
        raise ValueError(f"nothing to apply: {result.reason}")
    copy = Scorecard.from_config(scorecard.to_config())
    decision = copy.score(score_name).decision
    decision.model = result.head["model"]
    decision.classes = list(result.head["classes"])
    decision.weights = result.head["weights"]
    decision.positive_class = None
    decision.threshold = 0.0
    decision.calibration = result.calibration
    decision.provenance = result.provenance
    copy.validate()
    return copy


@dataclass
class Comparison:
    """A candidate against the incumbent, on the same labeled items and weights."""

    candidate: Summary
    incumbent: Summary
    promote: bool
    reasons: List[str]


def serve_summary(score: Score, questions: Mapping[str, Any], cache: AnswerCache,
                  training: TrainingSet) -> Summary:
    """How a score, served as it stands, does on the labeled items.

    The incumbent's own served predictions and calibration are used directly. It
    may have been fit on some of these same labels, which flatters it; the bias
    runs against promotion, so the comparison is conservative.
    """
    confidences: List[float] = []
    correct: List[int] = []
    for item_id, label in zip(training.item_ids, training.labels):
        result = predict(score, cache.partial_answers_for(item_id, questions))
        confidences.append(result.confidence or 0.0)
        correct.append(int(agrees(result.value, label)))
    return summarize(confidences, correct, training.weights)


def compare(candidate: FitResult, incumbent: Summary, *, min_brier_gain: float = 0.005,
            accuracy_tolerance: Optional[float] = None) -> Comparison:
    """Decide whether a candidate earns promotion.

    The candidate's numbers are out-of-fold, so they are honest; the incumbent's are
    direct. Brier is the primary criterion because it is a proper scoring rule that
    rewards accuracy and calibration together. A candidate that is merely better
    *calibrated* is still an improvement -- that is half the point -- and the reasons
    say which it was.

    Accuracy must not regress, but "not at all" would be the wrong bar. Accuracy is a
    step function: at 90 effective labels one item is 1.1 points, so a difference
    smaller than a couple of items is granularity, not evidence. The default
    tolerance is therefore two effective items, ``2 / n_effective``, which shrinks as
    labels accumulate and the comparison sharpens.
    """
    if not candidate.fitted or candidate.metrics is None:
        return Comparison(candidate.metrics or Summary(0, 0, 0, 0, 0), incumbent, False,
                          [candidate.reason or "no candidate was fitted"])
    if accuracy_tolerance is None:
        accuracy_tolerance = 2.0 / max(candidate.n_effective, 1.0)
    reasons: List[str] = []
    brier_gain = incumbent.brier - candidate.metrics.brier
    accuracy_change = candidate.metrics.accuracy - incumbent.accuracy
    if brier_gain < min_brier_gain:
        reasons.append(f"Brier improved by {brier_gain:+.4f}, below the {min_brier_gain} required")
    if accuracy_change < -accuracy_tolerance:
        reasons.append(f"accuracy fell by {-accuracy_change:.4f}, more than the "
                       f"{accuracy_tolerance:.4f} that {candidate.n_effective:.0f} effective "
                       "labels can resolve")
    return Comparison(candidate.metrics, incumbent, not reasons, reasons)
