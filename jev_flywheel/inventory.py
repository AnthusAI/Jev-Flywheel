"""How much each element actually matters, in a form an optimizer can trust.

The steering agent decides which elements to keep, reword or retire, so what it is
told about them has to be right. The obvious number, the fitted coefficient, is wrong
in two ways, and the lab notes have the receipt:

* **Coefficients live on different scales.** A log-odds feature ranges over roughly
  plus or minus 4.6; an expected-level feature over 0 to 1. A coefficient of 2 on the
  second is a far weaker statement than a coefficient of 2 on the first.
* **A large coefficient can be an offset, not a discovery.** In experiment 1 the
  intensity element carried the largest coefficient of all, yet its mean was nearly the
  same for positive and negative items within each tier. It marked which tier an item
  was in, and the model used it to rescale how much praise and criticism counted. An
  agent reading raw coefficients would have concluded intensity was the key concept.

So each element is reported with two numbers that do not have those problems:

* **Standardized weight** -- the coefficient times the feature's standard deviation, so
  it says how far the decision moves per one standard deviation of the input.
* **Permutation importance** -- how much the head's log loss *worsens* on the labeled
  items when that element's answers are shuffled across items. It is what the element
  is worth in practice, it needs no assumption about scale, and it handles the
  intensity case correctly: shuffling a tier marker leaves the loss nearly unchanged
  only if the model is not actually relying on it.

An element with permutation importance near zero is a candidate for retirement. One
with high importance is carrying the decision. Neither number is about *why*; that is
the language model's job, reading the mismatches and the human's comments.
"""
import random
from typing import Any, Dict, List, Mapping, Sequence

from jev_flywheel.features import split_feature
from jev_flywheel.models import REGISTRY, class_weights
from jev_flywheel.scorecard import Score


def _true_class_log_loss(rows: Sequence[Mapping[str, float]], labels: Sequence[str],
                         head: Mapping, weights: Sequence[float]) -> float:
    import math

    predictor = REGISTRY[head["model"]].predict_proba
    total = sum(weights)
    loss = 0.0
    for row, label, weight in zip(rows, labels, weights):
        p = predictor(row, head).get(label, 0.0)
        loss -= weight * math.log(max(p, 1e-9))
    return loss / total


def element_inventory(
    score: Score,
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[str],
    weights: Sequence[float],
    *,
    repeats: int = 8,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """One record per element (and the holistic answer), from the labeled items.

    Needs a decision with weights; an unfitted score has nothing to attribute, so it
    returns each element with zero importance and says so.
    """
    decision = score.decision
    if decision is None:
        return []
    head = decision.head()
    by_ref: Dict[str, List[str]] = {}
    for feature in decision.features:
        by_ref.setdefault(split_feature(feature)[0], []).append(feature)
    questions = {ref: (wire, spec) for ref, wire, spec in score.element_questions()}
    fitted = bool(class_weights(head)) and any(
        any(name != "intercept" for name in w) for w in class_weights(head).values())

    baseline = _true_class_log_loss(rows, labels, head, weights) if rows and fitted else 0.0
    weights_by_class = class_weights(head)
    rng = random.Random(seed)
    inventory: List[Dict[str, Any]] = []

    for ref, features in by_ref.items():
        std = {}
        for name in features:
            values = [row.get(name, 0.0) for row in rows]
            mean = sum(values) / len(values) if values else 0.0
            std[name] = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0
        # The largest absolute effect of any class's weight on this element's features.
        standardized = max(
            (abs(weights_by_class[c].get(name, 0.0)) * std[name]
             for c in weights_by_class for name in features), default=0.0)

        importance = 0.0
        if rows and fitted:
            gains = []
            for _ in range(repeats):
                order = list(range(len(rows)))
                rng.shuffle(order)
                shuffled = []
                for i, row in enumerate(rows):
                    replaced = dict(row)
                    for name in features:            # shuffle the element's features together
                        replaced[name] = rows[order[i]].get(name, 0.0)
                    shuffled.append(replaced)
                gains.append(_true_class_log_loss(shuffled, labels, head, weights) - baseline)
            importance = sum(gains) / len(gains)

        record: Dict[str, Any] = {
            "element": "holistic" if ref == "self.holistic" else ref,
            "features": features,
            "standardized_weight": round(standardized, 4),
            "permutation_importance": round(importance, 4),
        }
        if ref in questions:
            wire, spec = questions[ref]
            record.update({"question": spec.instructions, "type": spec.question_type})
        else:
            record.update({"question": score.instructions, "type": score.question_type})
        inventory.append(record)

    inventory.sort(key=lambda r: r["permutation_importance"], reverse=True)
    if not fitted:
        for record in inventory:
            record["note"] = "the head has no fitted weights yet, so importance is not available"
    return inventory
