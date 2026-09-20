"""Serving models: pure arithmetic over serialized numbers, standard library only.

Serving imports nothing but ``math``. Fitting may use scikit-learn and numpy
freely, but the fitted result is a plain dict of named weights that this module
can evaluate, which is what lets the same weights live in a YAML file, be read
by a human, and be diffed between versions.

Both models reduce to a softmax over per-class logits, so there is one serving
code path. ``linear_threshold`` is the hand-authorable two-class special case: it
exists so a stakeholder can write a rule before any labels exist at all.

Adding an architecture is one function plus one registry entry. Because features
are cached, comparing architectures costs no Jev calls.
"""
import math
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional

INTERCEPT = "intercept"


def class_weights(head: Mapping) -> Dict[str, Dict[str, float]]:
    """Per-class weight maps, with the hand-authored form expanded to the general one.

    For ``linear_threshold`` the threshold folds into the intercept and the
    negative class gets an empty map, which makes it a two-class softmax with
    all weight on one side.
    """
    classes: List[str] = list(head["classes"])
    if head["model"] == "linear_threshold":
        positive = head["positive_class"]
        flat = dict(head["weights"])
        flat[INTERCEPT] = flat.get(INTERCEPT, 0.0) - float(head.get("threshold", 0.0))
        return {c: (flat if c == positive else {}) for c in classes}
    weights = head.get("weights") or {}
    return {c: dict(weights.get(c, {})) for c in classes}


def declared_features(head: Mapping,
                      weights_by_class: Optional[Mapping[str, Dict[str, float]]] = None) -> List[str]:
    """Every feature name the weights mention, in first-seen order.

    ``weights_by_class`` lets a caller that already expanded the weights pass them in
    rather than have them rebuilt: expansion is cheap once and adds up when a whole
    pool is scored.
    """
    names: List[str] = []
    for weights in (weights_by_class or class_weights(head)).values():
        for name in weights:
            if name != INTERCEPT and name not in names:
                names.append(name)
    return names


def class_logits(features: Mapping[str, float], head: Mapping) -> Dict[str, float]:
    """One logit per class.

    A feature absent from ``features`` counts as 0. In log-odds space zero means
    "no evidence", so a degraded feature vector degrades gracefully instead of
    inventing a signal. Note that training refuses missing features -- see fit.
    """
    return {
        cls: weights.get(INTERCEPT, 0.0)
        + sum(w * features.get(name, 0.0) for name, w in weights.items() if name != INTERCEPT)
        for cls, weights in class_weights(head).items()
    }


def predict_proba(features: Mapping[str, float], head: Mapping) -> Dict[str, float]:
    """Softmax over the class logits, with the max subtracted for stability."""
    logits = class_logits(features, head)
    top = max(logits.values())
    exps = {c: math.exp(z - top) for c, z in logits.items()}
    total = sum(exps.values())
    return {c: e / total for c, e in exps.items()}


class ModelSpec(NamedTuple):
    """One architecture.

    ``min_n_effective`` is the floor the fitter refuses to go below. It is an
    effective sample size, not a row count: feedback sampled per confusion cell
    can carry weights an order of magnitude apart, so a handful of rows can
    dominate the likelihood.
    """

    name: str
    predict_proba: Callable[[Mapping[str, float], Mapping], Dict[str, float]]
    min_n_effective: float
    description: str


REGISTRY: Dict[str, ModelSpec] = {
    "linear_threshold": ModelSpec(
        name="linear_threshold",
        predict_proba=predict_proba,
        min_n_effective=1,
        description="Hand-authorable two-class rule. Needs no labels at all.",
    ),
    "multinomial_logistic": ModelSpec(
        name="multinomial_logistic",
        predict_proba=predict_proba,
        min_n_effective=50,
        description="Readable per-class weights. The default fitted head.",
    ),
}
