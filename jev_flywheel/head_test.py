"""Feature: the decision head turns named features into a value and a confidence."""
import math

import pytest

from jev_flywheel.head import decide, explain, validate_head
from jev_flywheel.models import REGISTRY


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


LOGISTIC = {
    "model": "multinomial_logistic",
    "classes": ["Yes", "No"],
    "weights": {
        "Yes": {"intercept": -1.0, "a.logit_p": 2.0, "b.logit_p": 1.0},
    },
}


def test_the_top_class_wins_and_its_probability_is_the_confidence():
    value, confidence, detail = decide({"a.logit_p": 1.0, "b.logit_p": 0.5}, LOGISTIC)

    assert value == "Yes"
    assert confidence == pytest.approx(sigmoid(-1.0 + 2.0 * 1.0 + 1.0 * 0.5))
    assert detail["probabilities"]["No"] == pytest.approx(1 - confidence)


def test_probabilities_sum_to_one():
    _, _, detail = decide({"a.logit_p": 3.0}, LOGISTIC)
    assert sum(detail["probabilities"].values()) == pytest.approx(1.0)


def test_a_missing_feature_contributes_nothing_and_lowers_coverage():
    # Zero in log-odds space means "no evidence", so a degraded vector degrades
    # gracefully rather than inventing a signal. Coverage records the damage.
    value, confidence, detail = decide({"a.logit_p": 1.0}, LOGISTIC)

    assert confidence == pytest.approx(sigmoid(-1.0 + 2.0))
    assert detail["coverage"] == 0.5


def test_full_coverage_when_every_declared_feature_is_present():
    _, _, detail = decide({"a.logit_p": 1.0, "b.logit_p": 1.0}, LOGISTIC)
    assert detail["coverage"] == 1.0


def test_contributions_name_the_biggest_driver_first():
    _, _, detail = decide({"a.logit_p": 0.1, "b.logit_p": 3.0}, LOGISTIC)

    names = [c["feature"] for c in detail["top_contributions"]]
    assert names[0] == "b.logit_p"


def test_a_contribution_is_the_weight_difference_times_the_feature():
    _, _, detail = decide({"a.logit_p": 2.0, "b.logit_p": 0.0}, LOGISTIC)

    top = next(c for c in detail["top_contributions"] if c["feature"] == "a.logit_p")
    # "No" carries no weights, so the difference is just the "Yes" weight.
    assert top["contribution"] == pytest.approx(2.0 * 2.0)


def test_the_explanation_names_the_value_the_confidence_and_the_drivers():
    value, confidence, detail = decide({"a.logit_p": 2.0, "b.logit_p": 1.0}, LOGISTIC)

    text = explain(value, confidence, detail)

    assert value in text
    assert "a.logit_p" in text
    assert "%" in text


def test_a_head_with_no_features_present_still_explains_itself():
    value, confidence, detail = decide({}, LOGISTIC)
    assert "none" in explain(value, confidence, detail)


def test_linear_threshold_is_a_hand_authorable_two_class_rule():
    # This exists so a stakeholder can write a rule before any labels exist.
    head = {"model": "linear_threshold", "classes": ["Pass", "Fail"],
            "positive_class": "Pass", "threshold": 0.5,
            "weights": {"intercept": 0.0, "a.p": 1.0}}

    assert decide({"a.p": 0.9}, head)[0] == "Pass"
    assert decide({"a.p": 0.1}, head)[0] == "Fail"


def test_the_abstain_band_flags_a_near_threshold_item_without_changing_its_value():
    head = {**LOGISTIC, "abstain_band": 0.1}

    near_value, _, near = decide({"a.logit_p": 0.5}, head)
    far_value, _, far = decide({"a.logit_p": 3.0}, head)

    assert near["abstain"] is True
    assert far["abstain"] is False
    # The flag is advisory. The value is whatever the model said.
    assert near_value == far_value == "Yes"


def test_validate_head_reports_every_problem_instead_of_raising():
    assert validate_head(LOGISTIC, features=["a.logit_p", "b.logit_p"]) == []
    assert any("model" in p for p in validate_head({**LOGISTIC, "model": "wat"}, features=[]))


def test_weights_for_a_class_that_does_not_exist_are_rejected():
    problems = validate_head({**LOGISTIC, "weights": {"Maybe": {"intercept": 0}}},
                             features=["a.logit_p"])
    assert any("Maybe" in p for p in problems)


def test_a_weight_on_an_undeclared_feature_is_rejected():
    problems = validate_head(LOGISTIC, features=["a.logit_p"])
    assert any("b.logit_p" in p for p in problems)


def test_a_head_needs_at_least_two_classes():
    assert any("two" in p for p in validate_head({**LOGISTIC, "classes": ["Yes"]}))


def test_linear_threshold_needs_a_positive_class_that_exists():
    head = {"model": "linear_threshold", "classes": ["Pass", "Fail"],
            "positive_class": "Nope", "weights": {}}
    assert any("positive_class" in p for p in validate_head(head))


def test_the_registry_lists_the_serving_models_with_sample_size_floors():
    assert {"linear_threshold", "multinomial_logistic"} <= set(REGISTRY)
    # A hand-written rule needs no labels; a fitted one does.
    assert REGISTRY["linear_threshold"].min_n_effective < REGISTRY["multinomial_logistic"].min_n_effective
