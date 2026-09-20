"""Confidence calibration: making "90% sure" mean 90% right.

A calibration is a monotone map from a raw confidence to a calibrated one. It is
stored as a 101-point lookup table, so applying it at serving time is
interpolation over a list of numbers -- no pickles, no scikit-learn, and a human
can read it in the YAML. Plexus stores its calibrations the same way
(``serialize_calibration_model``), so a table fit here is a table Plexus reads.

Three rules, each learned the expensive way:

1. **Out-of-fold only.** Isotonic regression is nonparametric and will memorize
   whatever it is fit on, producing a beautiful reliability diagram that means
   nothing. ``fit_calibration`` therefore accepts only ``OutOfFoldPredictions``,
   a type that fitting constructs from cross-validation and nothing else does.
   You cannot hand it in-sample predictions by accident.
2. **Match the method to the sample.** Temperature scaling has one parameter and
   is safe on a few dozen labels. Isotonic has as many parameters as there are
   distinct confidences, and on small samples it makes an already-calibrated
   model *worse* -- measured in this project's own lab notes. Two-stage
   (temperature, then isotonic) is for the label-rich end of the capability
   ladder. Below the floor, the honest answer is "no calibration".
3. **The head does part of the job already.** A logistic head with an intercept
   is calibrated on its training distribution by construction, so what is left
   for this layer is residual miscalibration, not the whole gap between Jev's
   raw confidence and the truth.
"""
import bisect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

GRID = 101
MIN_FOR_TEMPERATURE = 10
MIN_FOR_ISOTONIC = 50
METHODS = ("none", "temperature", "isotonic", "two_stage")


@dataclass(frozen=True)
class OutOfFoldPredictions:
    """Predictions made by a model that had not seen the item it predicted.

    Produced only by cross-validation in ``fit.py``. The type exists so that the
    calibration fitter cannot be handed anything else.
    """

    confidences: Sequence[float]
    correct: Sequence[int]
    weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        if len(self.confidences) != len(self.correct):
            raise ValueError("confidences and correct must be the same length")
        if self.weights is not None and len(self.weights) != len(self.confidences):
            raise ValueError("weights must match confidences in length")

    def __len__(self) -> int:
        return len(self.confidences)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def scale_temperature(confidence: float, temperature: float) -> float:
    """Sharpen (T < 1) or soften (T > 1) a top-label confidence."""
    return _sigmoid(_logit(confidence) / temperature)


def fit_temperature(predictions: OutOfFoldPredictions) -> float:
    """The temperature minimizing negative log-likelihood of being correct.

    Minimizes log loss rather than ECE. ECE is a step function of the
    temperature, so it has flat regions and local minima; log loss is smooth and
    has one minimum, which is what a one-parameter fit deserves.
    """
    from scipy.optimize import minimize_scalar

    w = list(predictions.weights) if predictions.weights is not None else [1.0] * len(predictions)
    logits = [_logit(c) for c in predictions.confidences]
    correct = list(predictions.correct)

    def loss(temperature: float) -> float:
        total = 0.0
        for wi, z, k in zip(w, logits, correct):
            p = min(max(_sigmoid(z / temperature), 1e-9), 1.0 - 1e-9)
            total -= wi * (k * math.log(p) + (1 - k) * math.log(1.0 - p))
        return total / sum(w)

    return float(minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded").x)


def fit_calibration(predictions: OutOfFoldPredictions, method: str) -> Dict[str, Any]:
    """Fit a calibration and return it as a serializable lookup table.

    Falls back to ``method: none`` when there is too little data for the
    requested method, and says why, because a calibration fit on too little data
    is worse than none.
    """
    if method not in METHODS:
        raise ValueError(f"unknown calibration method {method!r}; expected one of {METHODS}")
    n = len(predictions)
    need = {"none": 0, "temperature": MIN_FOR_TEMPERATURE,
            "isotonic": MIN_FOR_ISOTONIC, "two_stage": MIN_FOR_ISOTONIC}[method]
    if method == "none" or n < need:
        reason = None if method == "none" else (
            f"{method} needs at least {need} out-of-fold predictions and has {n}")
        return _identity(n, reason)

    temperature = fit_temperature(predictions) if method in ("temperature", "two_stage") else 1.0
    isotonic = None
    if method in ("isotonic", "two_stage"):
        from sklearn.isotonic import IsotonicRegression

        scaled = [scale_temperature(c, temperature) for c in predictions.confidences]
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(scaled, list(predictions.correct),
                     sample_weight=None if predictions.weights is None
                     else list(predictions.weights))

    raw = [i / (GRID - 1) for i in range(GRID)]
    calibrated: List[float] = []
    for value in raw:
        out = scale_temperature(value, temperature) if temperature != 1.0 else value
        if isotonic is not None:
            out = float(isotonic.predict([out])[0])
        calibrated.append(min(max(out, 0.0), 1.0))
    # Enforce monotonicity explicitly. Temperature scaling and isotonic are each
    # monotone, but the boundary clipping above could in principle break ties.
    for i in range(1, GRID):
        calibrated[i] = max(calibrated[i], calibrated[i - 1])

    return {
        "method": method,
        "temperature": temperature,
        "raw_confidence": raw,
        "calibrated_confidence": calibrated,
        "fit_on": "out_of_fold",
        "n": n,
    }


def _identity(n: int, reason: Optional[str]) -> Dict[str, Any]:
    raw = [i / (GRID - 1) for i in range(GRID)]
    out: Dict[str, Any] = {
        "method": "none", "temperature": 1.0,
        "raw_confidence": raw, "calibrated_confidence": list(raw),
        "fit_on": "out_of_fold", "n": n,
    }
    if reason:
        out["fallback_reason"] = reason
    return out


def apply_calibration(confidence: float, calibration: Optional[Mapping[str, Any]]) -> float:
    """Map a raw confidence through a stored calibration.

    Pure interpolation over the table, so serving needs nothing but this module's
    standard-library imports. ``None`` or ``method: none`` returns the input.
    """
    if not calibration or calibration.get("method", "none") == "none":
        return confidence
    xs = calibration["raw_confidence"]
    ys = calibration["calibrated_confidence"]
    if confidence <= xs[0]:
        return ys[0]
    if confidence >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_right(xs, confidence)
    lo = hi - 1
    span = xs[hi] - xs[lo]
    fraction = (confidence - xs[lo]) / span if span else 0.0
    return ys[lo] + fraction * (ys[hi] - ys[lo])
