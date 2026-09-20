"""From one item's Jev answers to a score result.

``ScoreResult`` mirrors Plexus's ``Score.Result``: a value, a confidence, an
explanation and a metadata dict. The fields and their meaning are deliberately the
same, so the console, the evaluator and the Tactus host all speak the vocabulary
Plexus does.

Two paths, chosen by whether the score declares a ``decision``:

* **No decision.** The result is Jev's holistic answer, passed through. This is
  what a Plexus JevScore does today, and it is the baseline everything is
  measured against.
* **A decision.** The result comes from the head over the score's features, and the
  confidence is calibrated. The holistic answer is just one feature among several,
  which is the whole architecture in one sentence.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from jev_flywheel.calibrate import apply_calibration
from jev_flywheel.head import decide, explain
from jev_flywheel.scorecard import Score, Scorecard


@dataclass
class ScoreResult:
    """One score's answer for one item. Same fields as Plexus's ``Score.Result``."""

    score_name: str
    value: str
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def raw_confidence(self) -> Optional[float]:
        """The head's confidence before calibration, when there was a decision."""
        detail = self.metadata.get("decision") or {}
        return detail.get("raw_confidence", self.confidence)


def _interpret(score: Score, answer: Mapping[str, Any]) -> tuple:
    """Jev's own holistic answer as ``(value, confidence, detail)``."""
    kind = score.question_type
    if kind == "noul":
        p = float(answer["noul"])
        return ("Yes" if p >= 0.5 else "No"), max(p, 1.0 - p), {"noul": p}
    if kind == "choice":
        choice = answer["choice"]
        probabilities = answer.get("probabilities") or {}
        return choice, float(probabilities.get(choice, answer.get("confidence") or 0.0)), \
            {"probabilities": dict(probabilities)}
    if kind == "score":
        return str(answer["score"]), float(answer.get("confidence") or 0.0), \
            {"legend": answer.get("legend")}
    raise ValueError(f"cannot interpret a {kind!r} answer")


def predict(score: Score, answers: Mapping[str, Any]) -> ScoreResult:
    """Score one item, given the answers to every question on the card.

    A missing element answer degrades gracefully -- its feature counts as zero and
    ``coverage`` in the metadata drops below 1 -- because a serving path that
    raised on a partial answer would turn one flaky request into an outage. Fitting
    is the opposite: it refuses missing features, since imputing them biases the
    weights.
    """
    decision = score.decision
    if decision is None:
        answer = answers.get(score.question_name)
        if answer is None:
            raise KeyError(
                f"no answer for {score.question_name!r} and the score has no decision to fall back on")
        value, confidence, detail = _interpret(score, answer)
        return ScoreResult(score.name, value, confidence, None, {"jev": detail})

    features = score.feature_vector(answers)
    value, raw, detail = decide(features, decision.head())
    calibrated = apply_calibration(raw, decision.calibration)
    detail = {**detail, "raw_confidence": raw}
    holistic = answers.get(score.question_name)
    metadata: Dict[str, Any] = {"decision": detail}
    if holistic is not None and score.question_type is not None:
        jev_value, jev_confidence, _ = _interpret(score, holistic)
        metadata["jev"] = {"value": jev_value, "confidence": jev_confidence}
    return ScoreResult(score.name, value, calibrated, explain(value, calibrated, detail), metadata)


def predict_scorecard(scorecard: Scorecard,
                      answers: Mapping[str, Any]) -> Dict[str, ScoreResult]:
    """Every score on the card, from one shared set of answers."""
    return {score.name: predict(score, answers) for score in scorecard.scores}
