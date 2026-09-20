"""The decision head at serving time: features in, value and confidence out.

The confidence is the probability of the value actually emitted, which is the
quantity a calibration step can correct and the quantity a reliability diagram
plots. It is deliberately a single scalar rather than a full distribution,
because that is what a score result carries and what a routing threshold needs.

The explanation is built deterministically from the largest contributions. That
matters more than it looks: reviewers react to explanations, and their reactions
are the training signal this whole system eats. A head that emits a bare number
starves its own feedback loop.
"""
from typing import Any, Dict, List, Mapping, Optional, Tuple

from jev_flywheel.models import REGISTRY, class_weights, declared_features

TOP_CONTRIBUTIONS = 3


def contributions(features: Mapping[str, float], head: Mapping, top: str,
                  runner_up: str, weights=None) -> Dict[str, float]:
    """How much each feature pushed the decision toward ``top`` and away from ``runner_up``.

    A contribution is the difference in the two classes' weights times the feature's
    value, so positive means "for the decision" and negative means "against it".
    Active selection reads the whole vector to measure how much of the evidence
    disagrees with the call the head made.
    """
    weights = weights or class_weights(head)
    return {
        name: (weights[top].get(name, 0.0) - weights[runner_up].get(name, 0.0))
        * features.get(name, 0.0)
        for name in declared_features(head, weights) if name in features
    }


def decide(features: Mapping[str, float], head: Mapping) -> Tuple[str, float, Dict[str, Any]]:
    """Return ``(value, confidence, detail)``, where confidence is P(value)."""
    probabilities = REGISTRY[head["model"]].predict_proba(features, head)
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    # Safe to index the runner-up because validate_head requires two classes.
    top, runner_up = ranked[0], ranked[1]

    weights = class_weights(head)
    declared = declared_features(head, weights)
    contributions_by_feature = contributions(features, head, top, runner_up, weights)
    contributions_ranked = [{"feature": name, "contribution": value}
                            for name, value in contributions_by_feature.items()]
    contributions_ranked.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    coverage = (sum(1 for name in declared if name in features) / len(declared)) if declared else 1.0
    band = float(head.get("abstain_band", 0.0))
    margin = (probabilities[top] - probabilities[runner_up]) / 2.0
    detail = {
        "model": head["model"],
        "probabilities": probabilities,
        "coverage": coverage,
        "top_contributions": contributions_ranked[:TOP_CONTRIBUTIONS],
        # A flag, never a change to the value. An abstaining item still gets an
        # answer; the flag says a human should look at it.
        "abstain": margin < band,
    }
    return top, probabilities[top], detail


def explain(value: str, confidence: float, detail: Mapping) -> str:
    """A short, deterministic account of why, from the largest contributions."""
    drivers = ", ".join(
        f"{c['feature']} ({c['contribution']:+.2f})" for c in detail["top_contributions"])
    return f"{value} at {confidence:.0%}. Largest drivers: {drivers or 'none'}."


def validate_head(head: Mapping, features: Optional[List[str]] = None) -> List[str]:
    """Problems with a decision head, as messages. Empty means valid.

    Returns messages rather than raising so a caller can report every problem in
    one pass, which is what a config linter wants.
    """
    problems: List[str] = []
    if head.get("model") not in REGISTRY:
        return [f"unknown model {head.get('model')!r}; expected one of {sorted(REGISTRY)}"]
    classes = head.get("classes") or []
    if len(classes) < 2 or not all(isinstance(c, str) for c in classes):
        return ["classes must list at least two label strings"]
    if head["model"] == "linear_threshold":
        if len(classes) != 2:
            problems.append("linear_threshold needs exactly two classes")
        if head.get("positive_class") not in classes:
            problems.append(f"positive_class {head.get('positive_class')!r} is not in classes")
        if problems:
            return problems
    else:
        for name in (head.get("weights") or {}):
            if name not in classes:
                problems.append(f"weights are given for {name!r}, which is not in classes")
        if problems:
            return problems
    if features is not None:
        for name in declared_features(head):
            if name not in features:
                problems.append(f"weight for {name!r} is not a declared feature")
    return problems
