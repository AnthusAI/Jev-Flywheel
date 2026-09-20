"""Feature: correcting the bias that active selection introduces.

The regression this guards against is the worst kind: a head that looks excellent
on the labeled items and does worse in production, with nothing raising.
"""
import random

import pytest

from jev_flywheel.sampling import (
    inverse_propensity_weights,
    kish_n_effective,
    selection_distribution,
)


def test_weights_are_the_reciprocal_of_the_propensity_up_to_scale():
    weights = inverse_propensity_weights([0.5, 0.25, 0.25], max_ratio=None)

    # 1/0.5 = 2, 1/0.25 = 4: the rarer pick counts twice as much.
    assert weights[1] == pytest.approx(2 * weights[0])
    assert weights[1] == pytest.approx(weights[2])


def test_weights_are_normalized_to_sum_to_the_item_count():
    # So a weighted fit is regularized as hard as an unweighted one.
    weights = inverse_propensity_weights([0.5, 0.1, 0.02, 0.9])

    assert sum(weights) == pytest.approx(4)


def test_a_very_unlikely_pick_is_capped_so_one_label_cannot_dominate_the_fit():
    weights = inverse_propensity_weights([0.5] * 9 + [0.0001], max_ratio=20, normalize=False)

    assert max(weights) == pytest.approx(20 * 2.0)   # 20 x the median weight of 2


def test_a_label_with_no_propensity_cannot_be_weighted_and_is_refused():
    # Guessing one would put a made-up number in the likelihood.
    with pytest.raises(ValueError, match="propensity"):
        inverse_propensity_weights([0.5, None])
    with pytest.raises(ValueError, match="propensity"):
        inverse_propensity_weights([0.5, 0.0])
    with pytest.raises(ValueError, match="propensity"):
        inverse_propensity_weights([1.5])


def test_equal_weights_give_an_effective_sample_size_equal_to_the_count():
    assert kish_n_effective([1.0] * 40) == pytest.approx(40)


def test_unequal_weights_shrink_the_effective_sample_size():
    # A handful of heavy rows dominate, so 12 rows are worth far fewer.
    weights = [50.0] + [1.0] * 11

    assert kish_n_effective(weights) < 3


def test_the_effective_size_is_never_larger_than_the_row_count():
    weights = [random.Random(1).uniform(0.1, 9) for _ in range(30)]
    assert kish_n_effective(weights) <= 30


def test_effective_size_of_nothing_is_zero():
    assert kish_n_effective([]) == 0.0


def test_a_selection_distribution_sums_to_one():
    assert sum(selection_distribution([3.0, 1.0, 0.0, -2.0])) == pytest.approx(1.0)


def test_a_higher_score_is_more_likely_to_be_shown():
    probabilities = selection_distribution([3.0, 1.0, 0.0])
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_every_item_keeps_a_strictly_positive_chance_however_low_it_scores():
    # Without the floor, a low-scoring item has propensity near zero and cannot be
    # inverse-weighted; with it, every item is reachable.
    probabilities = selection_distribution([100.0, 0.0, 0.0, 0.0], temperature=0.1, explore=0.2)

    assert min(probabilities) >= 0.2 / 4 - 1e-12


def test_the_explore_floor_bounds_the_largest_possible_weight():
    scores = [100.0] + [0.0] * 99
    probabilities = selection_distribution(scores, temperature=0.05, explore=0.1)

    assert max(1.0 / p for p in probabilities) <= len(scores) / 0.1 + 1e-9


def test_a_lower_temperature_concentrates_selection_on_the_best_item():
    warm = selection_distribution([2.0, 0.0], temperature=5.0, explore=0.05)
    cold = selection_distribution([2.0, 0.0], temperature=0.2, explore=0.05)

    assert cold[0] > warm[0]


def test_invalid_selection_settings_are_rejected():
    with pytest.raises(ValueError):
        selection_distribution([1.0], explore=0.0)
    with pytest.raises(ValueError):
        selection_distribution([1.0], temperature=0.0)


def test_weighting_recovers_the_population_rate_that_a_biased_sample_hides():
    # A population that is 90% negative. Selection strongly prefers positives, so
    # the labeled sample is badly skewed. Unweighted, the labeled positive rate is
    # far above the truth; inverse-propensity weighting brings it back.
    rng = random.Random(7)
    population = [1 if rng.random() < 0.10 else 0 for _ in range(20000)]
    pick = {1: 0.30, 0: 0.02}                        # positives are 15x likelier to be shown

    sample = [(y, pick[y]) for y in population if rng.random() < pick[y]]
    labels = [y for y, _ in sample]
    weights = inverse_propensity_weights([p for _, p in sample], max_ratio=None)

    unweighted = sum(labels) / len(labels)
    weighted = sum(w * y for w, y in zip(weights, labels)) / sum(weights)

    assert unweighted > 0.5                          # wildly biased
    assert weighted == pytest.approx(0.10, abs=0.02)  # recovered
