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
    twin_rows: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> TrainingSet:
    """Assemble what a fit needs, counting everything it had to leave out.

    ``needs_answers`` lists items that have a usable label but no complete set of
    cached answers -- typically because a new element was just proposed. They are
    left out of the fit and reported, so the caller can top up the cache and try
    again, rather than have the fit quietly train on fewer items than it appears to.

    ``twin_rows`` is ``studies/PREREGISTERED.md``'s L3 arm ("twin-augmented refit"): an
    optional ``{item_id: feature_row}`` map, one entry per labeled item, giving that item's
    *gender-swapped counterfactual twin*'s already-computed feature vector (``score.
    feature_vector`` on the twin's own answers -- computing those answers is the caller's job,
    since it needs an engine). When an item that made it into the training set has an entry
    here, its twin is appended as one more training row, carrying the *same* label and the
    *same* inverse-propensity weight as the original -- the twin is not an independent draw
    from the sampling design, so it must not get a weight of its own. ``None`` (the default)
    reproduces every existing caller's behaviour exactly.
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

        if twin_rows is not None and item_id in twin_rows:
            training.item_ids.append(f"{item_id}__twin")
            training.rows.append(dict(twin_rows[item_id]))
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
            accuracy_tolerance: Optional[float] = None,
            invariance_flip_rates: Optional[Mapping[str, float]] = None,
            max_flip_rate: float = 0.02) -> Comparison:
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

    ``invariance_flip_rates`` is the invariance gate from ``jev_flywheel.invariance``
    (``studies/PREREGISTERED.md``'s "does the engine read gender" section, arms J2/L2):
    ``None`` by default, which reproduces every existing caller's behaviour exactly. When
    given, it is ``{element_key: flip_rate}`` for each newly proposed element, measured on
    the labeled items' gender-swapped counterfactual twins; a candidate is rejected as a
    whole if any one new element flips on more than ``max_flip_rate`` of them, on top of
    (never instead of) the ordinary fit test above.
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
    if invariance_flip_rates is not None:
        from jev_flywheel.invariance import gate_new_elements

        gated = gate_new_elements(invariance_flip_rates, max_flip_rate=max_flip_rate)
        for key, result in gated.items():
            if not result.passed:
                reasons.append(f"element {key!r} failed the invariance gate: {result.reason}")
    return Comparison(candidate.metrics, incumbent, not reasons, reasons)


# ---------------------------------------------------------------------------------------------
# L4 -- the invariance penalty (studies/PREREGISTERED.md, "optimising the head against the
# flip"). A fit variant whose loss adds lambda * mean((P(surgeon|item) - P(surgeon|twin))^2)
# over the labeled pairs, on top of the ordinary weighted, L2-regularised logistic loss.
# sklearn's LogisticRegression cannot express a custom penalty term, so this is a small,
# hand-rolled binary logistic regression (numpy + scipy), used only behind this function --
# every other caller of fit_head is untouched.
# ---------------------------------------------------------------------------------------------

DEFAULT_LAMBDA_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


@dataclass
class LambdaPoint:
    """One point on the L4 lambda sweep: what a given penalty strength costs and buys.

    ``intercept``/``weights`` are the *full-data* (non-CV) fit at this point -- not used for
    any of the out-of-fold numbers above them, but reported so a reader can see the mechanism
    directly: the penalty term can only ever move ``weights`` and ``intercept`` together, and
    for a head with one feature, whether a labeled pair *flips* depends only on whether the
    item and its twin fall on opposite sides of the decision boundary (``weight * x +
    intercept == 0``). Shrinking ``weight`` while ``intercept`` barely moves relocates that
    boundary, but does not by itself change *which* pairs straddle it -- only a shift in
    ``intercept`` (relative to the spread of ``x``) does that.
    """

    lam: float
    oof_accuracy: float
    oof_log_loss: float
    oof_mean_abs_dp: float   # mean |P(item) - P(twin)| on the labeled pairs, out-of-fold
    intercept: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class InvarianceFitResult:
    """The full L4 sweep, plus the fitted head at the chosen operating point.

    ``points`` includes lambda = 0 first (the same objective ``fit_head`` optimises, at the
    same chosen C, refit here so it is directly comparable to the penalized points on the same
    CV splits) so "within 1 point of lambda=0" has a same-methodology baseline to compare
    against, not ``fit_head``'s own (differently split) out-of-fold number.
    """

    tier: Tier
    n: int
    n_effective: float
    chosen_c: float
    points: List[LambdaPoint]
    operating_lambda: float
    head: Dict[str, Any]
    provenance: Dict[str, Any]


def _sigmoid(z):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _penalized_binary_fit(X, y, w, c: float, lam: float, Xt=None, pair_mask=None,
                          max_iter: int = 500):
    """Fit a weighted, L2-regularised binary logistic regression with an optional invariance
    penalty, by direct minimisation (analytic gradient).

    ``y`` is 0/1 (class 1 is the "positive" class the returned coefficients score). ``w`` is
    per-row weight, already normalised to sum n (as ``sampling.inverse_propensity_weights``
    returns, and as sklearn's own ``sample_weight`` is used unmodified). ``Xt``/``pair_mask``
    are the twin feature matrix and a boolean mask over rows that have a twin (rows without one
    are still trained on ordinarily; they just contribute nothing to the penalty term). The
    objective, minimised over ``(intercept, coef)``, matches sklearn's ``LogisticRegression``
    parametrisation (``C`` scales the data term rather than dividing the regulariser, and the
    intercept is left unregularised) with the penalty term added on:

        c * sum_i w_i * log_loss_i  +  0.5 * ||coef||^2
                                     +  lam * mean_{i in pair_mask} (p_i - p_twin_i)^2
    """
    import numpy as np
    from scipy.optimize import minimize

    n, d = X.shape
    has_pairs = Xt is not None and pair_mask is not None and pair_mask.any()

    def objective(params):
        b, coef = params[0], params[1:]
        z = X @ coef + b
        p = _sigmoid(z)
        eps = 1e-9
        log_loss = c * float((w * -(y * np.log(np.clip(p, eps, 1))
                                    + (1 - y) * np.log(np.clip(1 - p, eps, 1)))).sum())
        grad_b = c * float((w * (p - y)).sum())
        grad_coef = c * (X.T @ (w * (p - y)))

        reg = 0.5 * float((coef ** 2).sum())
        grad_coef = grad_coef + coef

        penalty = 0.0
        if has_pairs:
            xi, xt = X[pair_mask], Xt[pair_mask]
            pi = _sigmoid(xi @ coef + b)
            pt = _sigmoid(xt @ coef + b)
            diff = pi - pt
            m = pair_mask.sum()
            penalty = lam * float((diff ** 2).mean())
            d_pi = pi * (1 - pi)
            d_pt = pt * (1 - pt)
            common = (2.0 * lam / m) * diff
            grad_b += float((common * (d_pi - d_pt)).sum())
            grad_coef = grad_coef + xi.T @ (common * d_pi) - xt.T @ (common * d_pt)

        loss = log_loss + reg + penalty
        grad = np.concatenate([[grad_b], grad_coef])
        return loss, grad

    x0 = np.zeros(d + 1)
    result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter})
    return result.x[0], result.x[1:]


def fit_head_invariance(
    training: TrainingSet,
    score: Score,
    twin_rows: Mapping[str, Mapping[str, float]],
    *,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    registered: Optional[Sequence[float]] = None,
    ladder: Tuple[Tier, ...] = CAPABILITY_LADDER_V1,
    folds: int = 5,
    seed: int = 0,
    accuracy_slack: float = 0.01,
) -> InvarianceFitResult:
    """L4: fit with the pairwise invariance penalty, swept over ``lambdas``.

    Only defined for a two-class decision (the penalty is stated in ``studies/
    PREREGISTERED.md`` as a difference of P(surgeon), a single number). ``twin_rows`` is
    ``{item_id: feature_row}`` for each labeled item's counterfactual twin -- exactly what
    ``build_training_set``'s own ``twin_rows`` argument takes for L3, but used here for the
    penalty rather than as extra rows.

    C is chosen once, via the same cross-validated, weighted-log-loss selection ``fit_head``
    uses (at lambda = 0, i.e. no penalty), and then reused for every lambda in the sweep --
    "the cross-validation that chooses C already exists and is reused" (the pre-registration).
    Every lambda is scored out-of-fold on the *same* CV splits, so the sweep is an
    apples-to-apples comparison. The operating point is the largest lambda whose out-of-fold
    accuracy is within ``accuracy_slack`` (one point, by default) of lambda = 0's.

    ``registered`` restricts which lambdas the operating point is chosen *from*, defaulting to
    ``lambdas`` itself (every existing caller's behaviour). Pass a subset when ``lambdas``
    carries extra, exploratory points beyond the pre-registered grid (``studies/
    PREREGISTERED.md``'s "optimising the head against the flip" section fixes the grid at
    {0.1, 1, 10, 100} in advance; an exploratory point past it must never move the operating
    point the pre-registration promised) -- every lambda in ``lambdas`` is still swept and
    reported, exploratory ones included.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    decision = score.decision
    if decision is None:
        raise ValueError(f"score {score.name!r} has no decision block")
    classes = sorted(set(training.labels))
    if len(classes) != 2:
        raise ValueError(f"the invariance penalty (L4) needs exactly two classes, got {classes}")
    # decision.classes gives the declared order; keep it if both are present, so the positive
    # class matches the convention the ordinary (sklearn) fit uses in ``_serving_head``.
    ordered = [c for c in decision.classes if c in classes] or classes
    positive = ordered[-1]
    negative = ordered[0]

    n, n_eff = training.n, training.n_effective
    tier = tier_for(n_eff, ladder)
    features = list(decision.features)
    X = build_matrix(training.rows, features)
    y = np.array([1 if label == positive else 0 for label in training.labels])
    w = np.array(training.weights, dtype=float)

    Xt = np.zeros_like(X)
    pair_mask = np.zeros(n, dtype=bool)
    for i, item_id in enumerate(training.item_ids):
        row = twin_rows.get(item_id)
        if row is not None:
            Xt[i] = [row.get(name, 0.0) for name in features]
            pair_mask[i] = True

    counts = Counter(training.labels)
    splitter = StratifiedKFold(n_splits=min(folds, min(counts.values())), shuffle=True,
                               random_state=seed)
    splits = list(splitter.split(X, y))

    # Reuse fit_head's own C selection: plain (lambda = 0) weighted log-loss CV over the tier's
    # C grid.
    def oof_proba_sklearn(c: float):
        out = np.zeros(n)
        for train, test in splits:
            model = LogisticRegression(C=c, max_iter=2000)
            model.fit(X[train], y[train], sample_weight=w[train])
            out[test] = model.predict_proba(X[test])[:, list(model.classes_).index(1)]
        return out

    def weighted_log_loss(p, y_, w_):
        eps = 1e-9
        return float(-(w_ * (y_ * np.log(np.clip(p, eps, 1)) + (1 - y_) * np.log(np.clip(1 - p, eps, 1)))).sum()
                     / w_.sum())

    c_grid = tier.c_grid or (1.0,)
    scored_c = {c: oof_proba_sklearn(c) for c in c_grid}
    chosen_c = min(scored_c, key=lambda c: weighted_log_loss(scored_c[c], y, w))

    def oof_for_lambda(lam: float):
        """One pass of the CV splits at this lambda: out-of-fold P(item) and, wherever the
        held-out item has a twin, out-of-fold P(twin) too (scored with the same fold's model,
        so it is exactly as out-of-fold as the item prediction it is compared against)."""
        proba = np.zeros(n)
        twin_pred = np.full(n, np.nan)
        for train, test in splits:
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train] = True
            fold_pair_mask = pair_mask & train_mask
            b, coef = _penalized_binary_fit(
                X[train], y[train], w[train], chosen_c, lam,
                Xt=Xt[train] if lam and fold_pair_mask.any() else None,
                pair_mask=fold_pair_mask[train] if lam else None)
            proba[test] = _sigmoid(X[test] @ coef + b)
            test = np.array(test)
            test_pairs = pair_mask[test]
            if test_pairs.any():
                twin_pred[test[test_pairs]] = _sigmoid(Xt[test[test_pairs]] @ coef + b)
        return proba, twin_pred

    points: List[LambdaPoint] = []
    full_lambda_grid = (0.0,) + tuple(lam for lam in lambdas if lam != 0.0)
    for lam in full_lambda_grid:
        proba, twin_pred = oof_for_lambda(lam)
        predicted = (proba >= 0.5).astype(int)
        acc = float((predicted == y).mean())
        ll = weighted_log_loss(proba, y, w)
        both = pair_mask & ~np.isnan(twin_pred)
        mean_abs_dp = float(np.abs(proba[both] - twin_pred[both]).mean()) if both.any() else 0.0
        # The full-data (non-CV) fit at this point too -- cheap at this sample size, and it is
        # what lets a reader see the mechanism (see LambdaPoint's docstring): does the penalty
        # move the intercept, or only the weight?
        b_point, coef_point = _penalized_binary_fit(
            X, y, w, chosen_c, lam, Xt=Xt if lam and pair_mask.any() else None,
            pair_mask=pair_mask if lam else None)
        points.append(LambdaPoint(
            lam=lam, oof_accuracy=acc, oof_log_loss=ll, oof_mean_abs_dp=mean_abs_dp,
            intercept=float(b_point),
            weights={name: float(v) for name, v in zip(features, coef_point)}))

    registered_set = set(registered if registered is not None else lambdas) | {0.0}
    base_accuracy = points[0].oof_accuracy
    candidates = [p for p in points if p.lam > 0 and p.lam in registered_set
                 and p.oof_accuracy >= base_accuracy - accuracy_slack]
    operating_lambda = max((p.lam for p in candidates), default=0.0)

    operating_point = next(p for p in points if p.lam == operating_lambda)
    head = {
        "model": "multinomial_logistic",
        "classes": list(decision.classes),
        "weights": {positive: {"intercept": operating_point.intercept,
                               **operating_point.weights}},
    }
    provenance = {
        "ladder": LADDER_NAME, "tier": tier.name, "n_train": n, "n_effective": round(n_eff, 2),
        "model": "multinomial_logistic", "regularization_c": chosen_c, "features": features,
        "arm": "L4", "lambda_grid": list(full_lambda_grid), "operating_lambda": operating_lambda,
        "accuracy_slack": accuracy_slack, "cv": {"folds": splitter.get_n_splits(), "seed": seed},
    }
    return InvarianceFitResult(tier=tier, n=n, n_effective=n_eff, chosen_c=chosen_c,
                               points=points, operating_lambda=operating_lambda, head=head,
                               provenance=provenance)
