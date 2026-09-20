"""Metrics: accuracy, calibration error, Brier score, log loss, reliability bins.

Pure Python, standard library only.

Two numbers must never be confused in this project, so this module is careful
about which population each function is handed:

* **Alignment** is agreement with the human on the items the human was shown.
  Those items were chosen by an active-selection rule, so they are a biased
  sample, and any metric over them has to be inverse-probability-weighted.
  Every function here therefore takes optional ``weights``.
* **Accuracy** is agreement with the corpus's own labels on the held-out test
  split, which no human sees and no selection rule touches. It needs no weights.

Confidence throughout means the probability of the predicted class, because that
is what a score result carries and what a routing threshold reads. Calibration
asks whether, among predictions made with about 90% confidence, about 90% were
right.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


def _weights(weights: Optional[Sequence[float]], n: int) -> List[float]:
    if weights is None:
        return [1.0] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} items")
    return [float(w) for w in weights]


def accuracy(correct: Sequence[int], weights: Optional[Sequence[float]] = None) -> float:
    """Weighted share of correct predictions."""
    w = _weights(weights, len(correct))
    total = sum(w)
    if total == 0:
        return 0.0
    return sum(wi * ci for wi, ci in zip(w, correct)) / total


@dataclass(frozen=True)
class Bin:
    """One bucket of a reliability diagram."""

    low: float
    high: float
    count: float
    mean_confidence: float
    accuracy: float


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[int], bins: int = 10,
    weights: Optional[Sequence[float]] = None,
) -> List[Bin]:
    """Equal-width bins of confidence against how often the prediction was right.

    This is the data behind a reliability diagram: a perfectly calibrated model
    puts every bin on the diagonal. Empty bins are omitted.
    """
    w = _weights(weights, len(confidences))
    members: List[List[int]] = [[] for _ in range(bins)]
    for index, confidence in enumerate(confidences):
        members[min(int(confidence * bins), bins - 1)].append(index)
    out: List[Bin] = []
    for number, indices in enumerate(members):
        mass = sum(w[i] for i in indices)
        if not indices or mass == 0:
            continue
        out.append(Bin(
            low=number / bins, high=(number + 1) / bins, count=mass,
            mean_confidence=sum(w[i] * confidences[i] for i in indices) / mass,
            accuracy=sum(w[i] * correct[i] for i in indices) / mass,
        ))
    return out


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[int], bins: int = 10,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Mass-weighted mean gap between confidence and accuracy over the bins.

    Its noise floor depends on sample size: at a few thousand items it is roughly
    0.01, and at a few dozen it is much larger, so a small ECE on a small set
    proves little.
    """
    if not confidences:
        return 0.0
    buckets = reliability_bins(confidences, correct, bins, weights)
    total = sum(b.count for b in buckets)
    if total == 0:
        return 0.0
    return sum(b.count / total * abs(b.mean_confidence - b.accuracy) for b in buckets)


def brier(confidences: Sequence[float], correct: Sequence[int],
          weights: Optional[Sequence[float]] = None) -> float:
    """Top-label Brier score: mean squared gap between confidence and correctness.

    Unlike ECE it rewards sharpness as well as calibration, so a model that says
    "60%" about everything cannot score well by being honest and useless.
    """
    if not confidences:
        return 0.0
    w = _weights(weights, len(confidences))
    total = sum(w)
    return sum(wi * (c - k) ** 2 for wi, c, k in zip(w, confidences, correct)) / total


def log_loss(true_class_probabilities: Sequence[float], eps: float = 1e-6,
             weights: Optional[Sequence[float]] = None) -> float:
    """Mean negative log probability assigned to the true class.

    The floor keeps one confident wrong answer from producing infinity: Jev often
    assigns exactly 0.0 or 1.0, which would otherwise make the score meaningless.
    """
    if not true_class_probabilities:
        return 0.0
    w = _weights(weights, len(true_class_probabilities))
    total = sum(w)
    return -sum(wi * math.log(max(p, eps))
                for wi, p in zip(w, true_class_probabilities)) / total


def confusion_matrix(predicted: Sequence[str], actual: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """``matrix[actual][predicted]`` counts, over every label seen on either side."""
    labels = sorted(set(predicted) | set(actual))
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for p, a in zip(predicted, actual):
        matrix[a][p] += 1
    return matrix


@dataclass(frozen=True)
class Summary:
    """The numbers every report prints."""

    n: int
    accuracy: float
    ece: float
    brier: float
    mean_confidence: float

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means the model is too sure."""
        return self.mean_confidence - self.accuracy


def summarize(confidences: Sequence[float], correct: Sequence[int],
              weights: Optional[Sequence[float]] = None) -> Summary:
    n = len(confidences)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0)
    w = _weights(weights, n)
    total = sum(w)
    return Summary(
        n=n,
        accuracy=accuracy(correct, w),
        ece=expected_calibration_error(confidences, correct, weights=w),
        brier=brier(confidences, correct, w),
        mean_confidence=sum(wi * c for wi, c in zip(w, confidences)) / total,
    )
