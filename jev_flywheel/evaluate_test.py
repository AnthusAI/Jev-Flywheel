"""Feature: metrics for accuracy and confidence calibration."""
import math

import pytest

from jev_flywheel.evaluate import (
    accuracy,
    brier,
    confusion_matrix,
    expected_calibration_error,
    log_loss,
    reliability_bins,
    summarize,
)


def test_accuracy_is_the_share_correct():
    assert accuracy([1, 1, 0, 1]) == 0.75


def test_a_weighted_accuracy_lets_heavy_items_count_more():
    # Actively-selected labels are biased toward hard items, so alignment has to
    # be inverse-probability-weighted or it lies.
    assert accuracy([1, 0], weights=[1, 3]) == pytest.approx(0.25)


def test_accuracy_of_nothing_is_zero_not_an_error():
    assert accuracy([]) == 0.0


def test_a_perfectly_calibrated_set_has_zero_calibration_error():
    # 10 predictions at 80% confidence, 8 of them right.
    confidences = [0.8] * 10
    correct = [1] * 8 + [0] * 2

    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0)


def test_an_overconfident_set_has_error_equal_to_the_gap():
    confidences = [0.95] * 10
    correct = [1] * 6 + [0] * 4

    assert expected_calibration_error(confidences, correct) == pytest.approx(0.35)


def test_calibration_error_is_weighted_by_how_many_items_land_in_each_bin():
    confidences = [0.9] * 8 + [0.5] * 2
    correct = [1] * 8 + [1, 0]           # first bin perfect, second bin 0.5 vs 0.5

    assert expected_calibration_error(confidences, correct) == pytest.approx(0.8 * 0.1)


def test_weights_change_calibration_error():
    confidences = [0.9, 0.9]
    correct = [1, 0]

    unweighted = expected_calibration_error(confidences, correct)
    weighted = expected_calibration_error(confidences, correct, weights=[9, 1])

    assert unweighted == pytest.approx(0.4)
    assert weighted == pytest.approx(0.0)


def test_a_confidence_of_exactly_one_lands_in_the_top_bin():
    bins = reliability_bins([1.0], [1], bins=10)
    assert bins[0].high == 1.0


def test_reliability_bins_omit_empty_buckets_and_report_each_bins_accuracy():
    bins = reliability_bins([0.15, 0.15, 0.95], [0, 1, 1], bins=10)

    assert [(b.low, b.count) for b in bins] == [(0.1, 2), (0.9, 1)]
    assert bins[0].accuracy == 0.5
    assert bins[0].mean_confidence == pytest.approx(0.15)


def test_brier_rewards_sharpness_that_ece_does_not():
    # Both are perfectly calibrated at 50% accuracy, but only one is informative.
    hedging = brier([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
    sharp = brier([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 0])

    assert hedging == pytest.approx(0.25)
    assert sharp == pytest.approx(0.5)     # sharp but wrong half the time is worse


def test_brier_of_a_perfect_confident_model_is_zero():
    assert brier([1.0, 1.0], [1, 1]) == 0.0


def test_log_loss_of_a_confident_correct_model_is_near_zero():
    assert log_loss([0.999, 0.999]) == pytest.approx(-math.log(0.999))


def test_log_loss_is_floored_so_one_certain_mistake_is_not_infinite():
    # Jev often assigns exactly 0.0 to the true class.
    assert math.isfinite(log_loss([0.0, 1.0]))
    assert log_loss([0.0]) == pytest.approx(-math.log(1e-6))


def test_a_confusion_matrix_counts_by_actual_then_predicted():
    matrix = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "b"])

    assert matrix == {"a": {"a": 1, "b": 0}, "b": {"a": 1, "b": 2}}


def test_a_confusion_matrix_includes_labels_seen_on_only_one_side():
    matrix = confusion_matrix(["a"], ["b"])
    assert set(matrix) == {"a", "b"}


def test_a_summary_reports_overconfidence_as_confidence_minus_accuracy():
    # The headline failure mode of raw Jev confidence: 0.913 mean confidence
    # against 0.760 accuracy.
    summary = summarize([0.9] * 10, [1] * 7 + [0] * 3)

    assert summary.n == 10
    assert summary.accuracy == pytest.approx(0.7)
    assert summary.mean_confidence == pytest.approx(0.9)
    assert summary.overconfidence == pytest.approx(0.2)


def test_summarizing_nothing_is_all_zeros():
    assert summarize([], []).n == 0


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError, match="weights"):
        accuracy([1, 0], weights=[1.0])
