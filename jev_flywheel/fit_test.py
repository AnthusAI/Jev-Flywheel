"""Feature: fitting the decision head from human feedback.

A synthetic world with a known truth: the label depends on two elements, and Jev's
holistic answer is a weak, noisy reading of it. So the specs can check that the fit
finds the structure, and can build deliberately biased samples to check the guards.
"""
import math
import random

import pytest

from jev_flywheel.answers import AnswerCache
from jev_flywheel.fit import (
    TrainingSet,
    build_matrix,
    build_training_set,
    compare,
    fit_head,
    latest_feedback,
    serve_summary,
    with_fit,
)
from jev_flywheel.head import decide
from jev_flywheel.items import (
    LABEL_SOURCE_FINAL,
    LABEL_SOURCE_SCORE_RESULT_OR_IMPORTED,
    LABEL_SOURCE_VETTED,
    FeedbackItem,
)
from jev_flywheel.ladder import LadderRefusal
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import predict

CARD = """
name: World
scores:
  - name: Outcome
    key: outcome
    question_type: noul
    instructions: "Good outcome?"
    elements:
      - {key: a, question_type: noul, instructions: "A?"}
      - {key: b, question_type: noul, instructions: "B?"}
      - {key: c, question_type: noul, instructions: "C?"}
    decision:
      model: multinomial_logistic
      classes: ["yes", "no"]
      features: [self.holistic.logit_p, a.logit_p, b.logit_p]
      parameters: {weights: {}}
"""


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def card():
    return Scorecard.from_yaml(CARD)


def signal(rng, truth, strength):
    """A noisy probability that leans toward the truth."""
    p = 0.5 + (0.5 if truth else -0.5) * strength + rng.gauss(0, 0.12)
    return min(max(p, 0.02), 0.98)


def world(n, seed=0, holistic=0.25, element=0.8):
    """``n`` items: truth, holistic answer (weak), elements a and b (strong), c (noise)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        truth = rng.random() < 0.5
        out.append((f"i{i}", "yes" if truth else "no", {
            "Outcome": {"type": "noul", "noul": signal(rng, truth, holistic)},
            "outcome.a": {"type": "noul", "noul": signal(rng, truth, element)},
            "outcome.b": {"type": "noul", "noul": signal(rng, truth, element)},
            "outcome.c": {"type": "noul", "noul": rng.uniform(0.05, 0.95)},
        }))
    return out


def load(cache, scorecard, data):
    questions = scorecard.questions()
    for item_id, _, answers in data:
        for name, answer in answers.items():
            cache.put(item_id, name, questions[name], answer)


def feedback_for(data, propensity=0.5, source=LABEL_SOURCE_FINAL):
    return [FeedbackItem(id=f"f-{item_id}", item_id=item_id, score_name="Outcome",
                         initial_answer_value="no", final_answer_value=label,
                         label_source=source, metadata={"propensity": propensity})
            for item_id, label, _ in data]


def training_from(n, seed=0, propensity=0.5, **kwargs):
    scorecard = card()
    data = world(n, seed, **kwargs)
    cache = AnswerCache()
    load(cache, scorecard, data)
    return build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache,
                              feedback_for(data, propensity)), scorecard


# ---- assembling the training set -------------------------------------------------

def test_a_training_set_holds_features_labels_and_weights_for_every_usable_item():
    training, scorecard = training_from(40)

    assert training.n == 40
    assert set(training.labels) <= {"yes", "no"}
    assert set(training.rows[0]) == set(scorecard.score("Outcome").decision.features)
    assert sum(training.weights) == pytest.approx(40)


def test_a_label_that_is_the_ais_own_prediction_is_never_trained_on():
    # The classic silent failure: it teaches the head to imitate the incumbent and
    # looks like it works because it agrees with the baseline.
    scorecard = card()
    data = world(6)
    cache = AnswerCache()
    load(cache, scorecard, data)
    feedback = feedback_for(data[:3]) + feedback_for(
        data[3:], source=LABEL_SOURCE_SCORE_RESULT_OR_IMPORTED)

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache, feedback)

    assert training.n == 3
    assert training.dropped == {"no_trusted_label": 3}


def test_vetted_labels_are_trusted_too():
    scorecard = card()
    data = world(4)
    cache = AnswerCache()
    load(cache, scorecard, data)

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache,
                                  feedback_for(data, source=LABEL_SOURCE_VETTED))

    assert training.n == 4


def test_an_invalidated_label_or_one_outside_the_classes_is_dropped_and_counted():
    scorecard = card()
    data = world(3)
    cache = AnswerCache()
    load(cache, scorecard, data)
    feedback = feedback_for(data)
    feedback[0].metadata["is_invalid"] = True
    feedback[1].final_answer_value = "maybe"

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache, feedback)

    assert training.n == 1
    assert training.dropped == {"no_trusted_label": 1, "label_not_in_classes": 1}


def test_a_label_with_no_recorded_propensity_is_dropped_not_guessed():
    scorecard = card()
    data = world(3)
    cache = AnswerCache()
    load(cache, scorecard, data)
    feedback = feedback_for(data)
    del feedback[0].metadata["propensity"]

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache, feedback)

    assert training.n == 2
    assert training.dropped == {"no_propensity": 1}


def test_a_default_propensity_lets_imported_labels_in_at_equal_weight():
    scorecard = card()
    data = world(3)
    cache = AnswerCache()
    load(cache, scorecard, data)
    feedback = feedback_for(data)
    for record in feedback:
        del record.metadata["propensity"]

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache,
                                  feedback, default_propensity=1.0)

    assert training.n == 3


def test_a_labeled_item_without_cached_answers_is_reported_for_a_top_up_not_trained_on():
    # Typically because a new element was just proposed. Silently training on
    # fewer items than it appears to would make the metrics misleading.
    scorecard = card()
    data = world(5)
    cache = AnswerCache()
    load(cache, scorecard, data[:3])

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache,
                                  feedback_for(data))

    assert training.n == 3
    assert sorted(training.needs_answers) == ["i3", "i4"]


def test_a_revisited_item_keeps_only_its_latest_label():
    first = FeedbackItem(id="1", item_id="a", score_name="Outcome", final_answer_value="yes")
    second = FeedbackItem(id="2", item_id="a", score_name="Outcome", final_answer_value="no")
    other = FeedbackItem(id="3", item_id="b", score_name="Elsewhere", final_answer_value="yes")

    latest = latest_feedback([first, second, other], "Outcome")

    assert list(latest) == ["a"]
    assert latest["a"].final_answer_value == "no"


def test_unequal_propensities_shrink_the_effective_sample_size():
    scorecard = card()
    data = world(40)
    cache = AnswerCache()
    load(cache, scorecard, data)
    feedback = feedback_for(data, propensity=0.5)
    for record in feedback[:5]:
        record.metadata["propensity"] = 0.01

    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache, feedback)

    assert training.n_effective < training.n


def test_a_score_with_no_decision_cannot_be_fitted():
    plain = Scorecard.from_yaml('name: C\nscores:\n  - {name: Q, question_type: noul, instructions: "?"}\n')

    with pytest.raises(ValueError, match="no decision"):
        build_training_set(plain.score("Q"), plain.questions(), AnswerCache(), [])


def test_training_refuses_a_row_with_a_missing_feature_where_serving_would_impute_zero():
    # Imputing neutral rows biases a new element's weight toward zero, and then the
    # optimizer concludes its own proposal was useless and retires it.
    with pytest.raises(ValueError, match="full coverage"):
        build_matrix([{"a.logit_p": 1.0}], ["a.logit_p", "b.logit_p"])


# ---- fitting -----------------------------------------------------------------------

def test_too_few_labels_hold_the_incumbent_rather_than_fitting_noise():
    training, scorecard = training_from(15)

    result = fit_head(training, scorecard.score("Outcome"))

    assert result.status == "held"
    assert not result.fitted
    assert result.head is None
    assert "more are needed" in result.reason


def test_enough_labels_produce_a_fitted_head_and_say_which_tier_allowed_it():
    training, scorecard = training_from(120)

    result = fit_head(training, scorecard.score("Outcome"))

    assert result.fitted
    assert result.tier.name == "shrunk"
    assert result.provenance["ladder"] == "capability-ladder-v1"
    assert result.provenance["n_effective"] == pytest.approx(120, abs=0.5)


def test_the_fit_finds_the_informative_elements_and_ignores_none_it_was_not_given():
    training, scorecard = training_from(300)

    result = fit_head(training, scorecard.score("Outcome"))
    weights = result.head["weights"]["yes"]

    # a and b carry the signal; the holistic answer is weak.
    assert weights["a.logit_p"] > 0 and weights["b.logit_p"] > 0
    assert weights["a.logit_p"] > weights["self.holistic.logit_p"]


def test_a_fitted_head_beats_jevs_own_holistic_answer_on_a_world_where_elements_matter():
    training, scorecard = training_from(400)
    result = fit_head(training, scorecard.score("Outcome"))

    holistic_only = sum(
        (row["self.holistic.logit_p"] > 0) == (label == "yes")
        for row, label in zip(training.rows, training.labels)) / training.n

    assert result.metrics.accuracy > holistic_only + 0.05


def test_the_served_head_matches_scikit_learns_probabilities_exactly():
    # The fitted head serves through stdlib-only code. If it disagreed with the
    # library that fit it, every number reported here would be suspect.
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    training, scorecard = training_from(200)
    result = fit_head(training, scorecard.score("Outcome"))
    features = scorecard.score("Outcome").decision.features
    X = build_matrix(training.rows, features)
    y = np.array([1 if label == "yes" else 0 for label in training.labels])
    reference = LogisticRegression(C=result.chosen_c, max_iter=2000)
    reference.fit(X, y, sample_weight=np.array(training.weights))

    for row, expected in zip(training.rows[:25], reference.predict_proba(X[:25])):
        served = decide(row, result.head)[2]["probabilities"]
        assert served["yes"] == pytest.approx(expected[1], abs=1e-9)


def test_calibration_is_fit_on_out_of_fold_predictions():
    training, scorecard = training_from(300)

    result = fit_head(training, scorecard.score("Outcome"))

    assert result.calibration["fit_on"] == "out_of_fold"
    assert len(result.oof) == training.n


def test_more_effective_labels_unlock_a_richer_calibration_method():
    small, scorecard = training_from(80)
    large, _ = training_from(1100)

    assert fit_head(small, scorecard.score("Outcome")).calibration["method"] == "temperature"
    assert fit_head(large, scorecard.score("Outcome")).calibration["method"] == "two_stage"


def test_asking_for_more_features_than_the_evidence_supports_is_refused_with_a_way_out():
    many = CARD.replace("[self.holistic.logit_p, a.logit_p, b.logit_p]",
                        "[self.holistic.logit_p, a.logit_p, b.logit_p, c.logit_p, a.p, b.p, c.p, a.is_yes, b.is_yes]")
    scorecard = Scorecard.from_yaml(many)
    data = world(35)
    cache = AnswerCache()
    load(cache, scorecard, data)
    training = build_training_set(scorecard.score("Outcome"), scorecard.questions(), cache,
                                  feedback_for(data))

    with pytest.raises(LadderRefusal, match="supports 7"):
        fit_head(training, scorecard.score("Outcome"))


def test_a_single_class_of_labels_is_refused():
    training, scorecard = training_from(60)
    training.labels = ["yes"] * training.n

    with pytest.raises(LadderRefusal, match="both classes"):
        fit_head(training, scorecard.score("Outcome"))


def test_a_class_with_almost_no_labels_is_refused():
    training, scorecard = training_from(60)
    training.labels = ["yes"] * (training.n - 2) + ["no"] * 2

    with pytest.raises(LadderRefusal, match="at least 3"):
        fit_head(training, scorecard.score("Outcome"))


def test_provenance_records_what_a_reviewer_needs_to_reproduce_the_fit():
    training, scorecard = training_from(120)

    provenance = fit_head(training, scorecard.score("Outcome")).provenance

    for key in ("fit_id", "tier", "n_train", "n_effective", "regularization_c", "features",
                "question_set_fingerprint", "cv", "label_prior_population", "cell_census"):
        assert key in provenance
    assert provenance["fit_id"].startswith("dh-")
    assert provenance["cell_census"] == {"no->yes": provenance["cell_census"]["no->yes"],
                                         "no->no": provenance["cell_census"]["no->no"]}


def test_the_fit_is_deterministic_for_a_fixed_seed():
    training, scorecard = training_from(150)
    first = fit_head(training, scorecard.score("Outcome"), seed=3)
    second = fit_head(training, scorecard.score("Outcome"), seed=3)

    assert first.head == second.head
    assert first.provenance["fit_id"] == second.provenance["fit_id"]


# ---- the sampling correction -----------------------------------------------------

def test_weighting_recovers_the_population_rate_that_a_biased_sample_hides():
    """Selection strongly prefers one class. Unweighted, the head thinks it is common.

    The population is 10% "yes". Positives are 15 times likelier to be shown, so
    the labeled sample is mostly "yes". A head fit on it unweighted predicts "yes"
    far too often; weighting by the recorded propensity recovers the true base rate.
    """
    rng = random.Random(11)
    scorecard = card()
    score = scorecard.score("Outcome")
    pick = {"yes": 0.30, "no": 0.02}

    def sample(n):
        rows, labels, props = [], [], []
        while len(rows) < n:
            label = "yes" if rng.random() < 0.10 else "no"
            if rng.random() < pick[label]:
                # A weak feature: the base rate has to do most of the work.
                z = (0.3 if label == "yes" else -0.3) + rng.gauss(0, 1.0)
                rows.append({"self.holistic.logit_p": z, "a.logit_p": rng.gauss(0, 1),
                             "b.logit_p": rng.gauss(0, 1)})
                labels.append(label)
                props.append(pick[label])
        return rows, labels, props

    rows, labels, props = sample(600)
    from jev_flywheel.sampling import inverse_propensity_weights

    def fitted(weights):
        training = TrainingSet(
            item_ids=[str(i) for i in range(len(rows))], rows=rows, labels=labels,
            weights=weights, cells=[None] * len(rows))
        return fit_head(training, score)

    weighted = fitted(inverse_propensity_weights(props, max_ratio=None))
    unweighted = fitted([1.0] * len(rows))

    population_rows = [{"self.holistic.logit_p": (0.3 if rng.random() < 0.10 else -0.3) + rng.gauss(0, 1),
                        "a.logit_p": rng.gauss(0, 1), "b.logit_p": rng.gauss(0, 1)}
                       for _ in range(4000)]

    def mean_yes(result):
        return sum(decide(r, result.head)[2]["probabilities"]["yes"]
                   for r in population_rows) / len(population_rows)

    assert mean_yes(unweighted) > 0.35          # wildly over-predicts the rare class
    assert mean_yes(weighted) < 0.25            # closer to the true 0.10


# ---- applying and comparing ------------------------------------------------------

def test_applying_a_fit_changes_the_numbers_and_leaves_the_words_alone():
    training, scorecard = training_from(200)
    result = fit_head(training, scorecard.score("Outcome"))

    updated = with_fit(scorecard, "Outcome", result)

    before, after = scorecard.score("Outcome"), updated.score("Outcome")
    assert after.decision.weights == result.head["weights"]
    assert after.decision.calibration == result.calibration
    assert after.decision.provenance["fit_id"] == result.provenance["fit_id"]
    assert [e.to_config() for e in after.elements] == [e.to_config() for e in before.elements]
    assert before.decision.weights == {}                 # the original is untouched


def test_an_updated_scorecard_round_trips_through_yaml_and_serves():
    training, scorecard = training_from(200)
    updated = with_fit(scorecard, "Outcome", fit_head(training, scorecard.score("Outcome")))

    reloaded = Scorecard.from_yaml(updated.to_yaml())
    answers = world(1, seed=99)[0][2]
    result = predict(reloaded.score("Outcome"), answers)

    assert result.value in {"yes", "no"}
    assert 0.0 <= result.confidence <= 1.0


def test_a_held_fit_cannot_be_applied():
    training, scorecard = training_from(10)
    result = fit_head(training, scorecard.score("Outcome"))

    with pytest.raises(ValueError, match="nothing to apply"):
        with_fit(scorecard, "Outcome", result)


def test_a_candidate_that_is_better_calibrated_and_more_accurate_is_promoted():
    training, scorecard = training_from(300)
    candidate = fit_head(training, scorecard.score("Outcome"))
    # The incumbent is the raw holistic answer: overconfident and weak.
    plain = Scorecard.from_yaml(
        'name: C\nscores:\n  - {name: Outcome, key: outcome, question_type: noul, instructions: "Good outcome?"}\n')
    cache = AnswerCache()
    load(cache, scorecard, world(300))
    incumbent = serve_summary(plain.score("Outcome"), plain.questions(), cache, training)

    comparison = compare(candidate, incumbent)

    assert comparison.promote
    assert comparison.reasons == []
    assert comparison.candidate.brier < comparison.incumbent.brier


def test_a_candidate_that_does_not_beat_the_incumbent_is_rejected_with_the_reason():
    # A noisy world, so the candidate cannot itself be perfect and an incumbent
    # claiming perfection is genuinely ahead on both counts.
    training, scorecard = training_from(200, element=0.3, holistic=0.1)
    candidate = fit_head(training, scorecard.score("Outcome"))
    assert candidate.metrics.accuracy < 1.0
    from jev_flywheel.evaluate import Summary
    perfect = Summary(n=200, accuracy=1.0, ece=0.0, brier=0.0, mean_confidence=1.0)

    comparison = compare(candidate, perfect)

    assert not comparison.promote
    assert any("Brier" in r for r in comparison.reasons)
    assert any("accuracy" in r for r in comparison.reasons)


def test_a_held_candidate_is_never_promoted():
    training, scorecard = training_from(10)
    from jev_flywheel.evaluate import Summary

    comparison = compare(fit_head(training, scorecard.score("Outcome")),
                         Summary(10, 0.5, 0.1, 0.3, 0.6))

    assert not comparison.promote
    assert "more are needed" in comparison.reasons[0]


def _candidate(accuracy, brier, n_effective):
    from jev_flywheel.evaluate import Summary
    from jev_flywheel.fit import FitResult
    from jev_flywheel.ladder import tier_for
    return FitResult("fitted", tier_for(n_effective), int(n_effective), n_effective,
                     metrics=Summary(int(n_effective), accuracy, 0.02, brier, accuracy))


def test_a_one_item_dip_in_accuracy_is_noise_not_a_regression_at_small_samples():
    # At 90 effective labels one item is 1.1 points. A better-calibrated candidate
    # that is one item less accurate should still be promoted.
    from jev_flywheel.evaluate import Summary

    incumbent = Summary(90, accuracy=0.745, ece=0.17, brier=0.20, mean_confidence=0.9)

    comparison = compare(_candidate(accuracy=0.733, brier=0.183, n_effective=90), incumbent)

    assert comparison.promote, comparison.reasons


def test_a_real_accuracy_regression_is_still_rejected_however_good_the_calibration():
    from jev_flywheel.evaluate import Summary

    incumbent = Summary(90, accuracy=0.80, ece=0.17, brier=0.20, mean_confidence=0.9)

    comparison = compare(_candidate(accuracy=0.70, brier=0.15, n_effective=90), incumbent)

    assert not comparison.promote
    assert "accuracy fell" in comparison.reasons[0]


def test_the_accuracy_tolerance_tightens_as_labels_accumulate():
    from jev_flywheel.evaluate import Summary

    incumbent = Summary(1000, accuracy=0.80, ece=0.1, brier=0.20, mean_confidence=0.8)
    dip = _candidate(accuracy=0.788, brier=0.15, n_effective=1000)   # 1.2 points down

    # Two effective items is 0.2 points at 1000 labels, so 1.2 points is real.
    assert not compare(dip, incumbent).promote
