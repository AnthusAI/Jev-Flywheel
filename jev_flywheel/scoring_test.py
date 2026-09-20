"""Feature: scoring an item from its Jev answers."""
import math

import pytest

from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import predict, predict_scorecard

CARD = """
name: Card
scores:
  - name: Escalate
    key: escalate
    question_type: noul
    instructions: "Escalate?"
    elements:
      - {key: angry, question_type: noul, instructions: "Angry?"}
    decision:
      model: multinomial_logistic
      classes: ["Yes", "No"]
      features: [self.holistic.logit_p, angry.logit_p]
      parameters:
        weights:
          "Yes": {intercept: -0.5, self.holistic.logit_p: 1.0, angry.logit_p: 0.5}
  - name: Tone
    key: tone
    question_type: choice
    instructions: "Tone?"
    criteria: {calm: null, angry: null}
"""

ANSWERS = {
    "Escalate": {"type": "noul", "noul": 0.7},
    "escalate.angry": {"type": "noul", "noul": 0.9},
    "Tone": {"type": "choice", "choice": "calm", "confidence": 0.8,
             "probabilities": {"calm": 0.8, "angry": 0.2}},
}


def logit(p):
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def card():
    return Scorecard.from_yaml(CARD)


def test_a_score_without_a_decision_passes_jevs_holistic_answer_through():
    result = predict(card().score("Tone"), ANSWERS)

    assert result.value == "calm"
    assert result.confidence == pytest.approx(0.8)
    assert result.explanation is None


def test_a_yes_no_holistic_answer_becomes_yes_or_no_with_the_larger_probability():
    plain = Scorecard.from_yaml(
        'name: C\nscores:\n  - {name: Q, question_type: noul, instructions: "Q?"}\n')

    yes = predict(plain.score("Q"), {"Q": {"noul": 0.7}})
    no = predict(plain.score("Q"), {"Q": {"noul": 0.2}})

    assert (yes.value, yes.confidence) == ("Yes", pytest.approx(0.7))
    assert (no.value, no.confidence) == ("No", pytest.approx(0.8))


def test_a_score_with_a_decision_combines_holistic_answer_and_elements():
    result = predict(card().score("Escalate"), ANSWERS)

    z = -0.5 + logit(0.7) + 0.5 * logit(0.9)
    assert result.value == "Yes"
    assert result.confidence == pytest.approx(sigmoid(z))
    assert result.metadata["decision"]["coverage"] == 1.0


def test_the_explanation_names_the_features_that_drove_the_decision():
    result = predict(card().score("Escalate"), ANSWERS)

    assert "Yes" in result.explanation
    assert "angry.logit_p" in result.explanation or "self.holistic.logit_p" in result.explanation


def test_the_result_keeps_jevs_own_answer_beside_the_decision_for_comparison():
    result = predict(card().score("Escalate"), ANSWERS)

    assert result.metadata["jev"]["value"] == "Yes"
    assert result.metadata["jev"]["confidence"] == pytest.approx(0.7)


def test_the_decision_can_overrule_the_holistic_answer():
    # Jev says No (0.3), but the elements are strong enough to win.
    answers = {**ANSWERS, "Escalate": {"noul": 0.3}, "escalate.angry": {"noul": 0.99}}
    heavy = CARD.replace("angry.logit_p: 0.5", "angry.logit_p: 2.0")
    score = Scorecard.from_yaml(heavy).score("Escalate")

    result = predict(score, answers)

    assert result.metadata["jev"]["value"] == "No"
    assert result.value == "Yes"


def test_a_missing_element_answer_degrades_to_zero_and_lowers_coverage():
    # Serving must not turn one flaky request into an outage.
    answers = {k: v for k, v in ANSWERS.items() if k != "escalate.angry"}

    result = predict(card().score("Escalate"), answers)

    assert result.confidence == pytest.approx(sigmoid(-0.5 + logit(0.7)))
    assert result.metadata["decision"]["coverage"] == 0.5


def test_a_calibration_is_applied_to_the_confidence_but_the_raw_value_is_kept():
    text = CARD.replace(
        "      parameters:\n        weights:\n          \"Yes\"",
        "      calibration:\n        method: temperature\n        raw_confidence: [0.0, 1.0]\n"
        "        calibrated_confidence: [0.0, 0.5]\n"
        "      parameters:\n        weights:\n          \"Yes\"")
    score = Scorecard.from_yaml(text).score("Escalate")

    result = predict(score, ANSWERS)

    raw = sigmoid(-0.5 + logit(0.7) + 0.5 * logit(0.9))
    assert result.confidence == pytest.approx(raw * 0.5)
    assert result.raw_confidence == pytest.approx(raw)
    assert result.value == "Yes"        # calibration never changes the decision


def test_a_score_with_no_decision_and_no_answer_says_so():
    with pytest.raises(KeyError, match="Tone"):
        predict(card().score("Tone"), {})


def test_a_whole_scorecard_is_scored_from_one_shared_set_of_answers():
    results = predict_scorecard(card(), ANSWERS)

    assert set(results) == {"Escalate", "Tone"}
    assert results["Tone"].value == "calm"
