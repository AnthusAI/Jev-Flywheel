"""Feature: an honest account of which elements matter.

The trap this guards against: reading raw coefficients as importances. A feature on a
wide scale, or one that merely marks a subgroup, can carry the biggest coefficient
without carrying the decision.
"""
import random

import pytest

from jev_flywheel.inventory import element_inventory
from jev_flywheel.scorecard import Scorecard

CARD = """
name: World
scores:
  - name: Outcome
    key: outcome
    question_type: noul
    instructions: "Good?"
    elements:
      - {key: signal, question_type: noul, instructions: "The one that matters?"}
      - {key: noise, question_type: noul, instructions: "Unrelated?"}
      - {key: tiny_scale, question_type: noul, instructions: "Right but narrow?"}
    decision:
      model: multinomial_logistic
      classes: ["yes", "no"]
      features: [self.holistic.logit_p, signal.logit_p, noise.logit_p, tiny_scale.p]
      parameters:
        weights:
          "yes": {intercept: 0.0, self.holistic.logit_p: 0.0, signal.logit_p: 1.5,
                  noise.logit_p: 1.5, tiny_scale.p: 1.5}
"""


def score():
    return Scorecard.from_yaml(CARD).score("Outcome")


def world(n=600, seed=0):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        truth = rng.random() < 0.5
        rows.append({
            "self.holistic.logit_p": rng.gauss(0, 1),
            "signal.logit_p": rng.gauss(2.0 if truth else -2.0, 1.0),   # informative
            "noise.logit_p": rng.gauss(0, 2.0),                          # wide scale, no signal
            "tiny_scale.p": 0.5 + (0.01 if truth else -0.01) + rng.gauss(0, 0.01),
        })
        labels.append("yes" if truth else "no")
    return rows, labels, [1.0] * n


def by_element(inventory):
    return {r["element"]: r for r in inventory}


def test_an_element_that_carries_the_decision_has_high_permutation_importance():
    rows, labels, weights = world()

    inventory = by_element(element_inventory(score(), rows, labels, weights))

    assert inventory["signal"]["permutation_importance"] > 0.3


def test_an_element_with_no_signal_has_near_zero_importance_despite_an_equal_coefficient():
    # Identical coefficients (1.5), very different worth. This is the point.
    rows, labels, weights = world()

    inventory = by_element(element_inventory(score(), rows, labels, weights))

    assert inventory["noise"]["permutation_importance"] < inventory["signal"]["permutation_importance"] / 3


def test_a_wide_scale_feature_looks_important_by_standardized_weight_but_is_not():
    # Standardized weight alone still flatters a wide-scale noise feature, which is why
    # permutation importance is reported beside it and is the one to act on.
    rows, labels, weights = world()

    inventory = by_element(element_inventory(score(), rows, labels, weights))

    assert inventory["noise"]["standardized_weight"] > inventory["tiny_scale"]["standardized_weight"]
    assert inventory["noise"]["permutation_importance"] < inventory["signal"]["permutation_importance"]


def test_the_result_is_sorted_most_important_first_so_the_top_is_what_to_protect():
    rows, labels, weights = world()

    inventory = element_inventory(score(), rows, labels, weights)

    importances = [r["permutation_importance"] for r in inventory]
    assert importances == sorted(importances, reverse=True)
    assert inventory[0]["element"] == "signal"


def test_each_record_names_the_element_its_features_and_its_question():
    rows, labels, weights = world()

    record = by_element(element_inventory(score(), rows, labels, weights))["signal"]

    assert record["features"] == ["signal.logit_p"]
    assert record["question"] == "The one that matters?"
    assert record["type"] == "noul"


def test_the_holistic_answer_is_reported_like_any_element():
    rows, labels, weights = world()

    inventory = by_element(element_inventory(score(), rows, labels, weights))

    assert "holistic" in inventory
    assert inventory["holistic"]["question"] == "Good?"


def test_the_inventory_is_deterministic_for_a_fixed_seed():
    rows, labels, weights = world()

    first = element_inventory(score(), rows, labels, weights, seed=4)
    second = element_inventory(score(), rows, labels, weights, seed=4)

    assert first == second


def test_an_unfitted_head_says_importance_is_not_available_rather_than_reporting_zeros():
    unfitted = CARD.replace(
        '"yes": {intercept: 0.0, self.holistic.logit_p: 0.0, signal.logit_p: 1.5,\n'
        '                  noise.logit_p: 1.5, tiny_scale.p: 1.5}', '"yes": {intercept: 0.0}')
    rows, labels, weights = world(100)

    inventory = element_inventory(Scorecard.from_yaml(unfitted).score("Outcome"),
                                  rows, labels, weights)

    assert all("not available" in r["note"] for r in inventory)


def test_a_score_with_no_decision_has_no_inventory():
    plain = Scorecard.from_yaml('name: C\nscores:\n  - {name: Q, question_type: noul, instructions: "?"}\n')

    assert element_inventory(plain.score("Q"), [], [], []) == []


def test_a_marker_that_the_head_does_not_rely_on_is_not_reported_as_the_key_concept():
    # The intensity trap from the lab notes: it looks big, but shuffling it barely
    # changes the loss because the decision is really carried by the signal.
    rows, labels, weights = world()
    inventory = by_element(element_inventory(score(), rows, labels, weights))

    assert inventory["signal"]["permutation_importance"] > 5 * inventory["noise"]["permutation_importance"]
