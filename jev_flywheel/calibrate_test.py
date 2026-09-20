"""Feature: calibrating confidence.

Uses a synthetic model that is overconfident in a known way, so the right answer
is known and the specs can check the calibration finds it.
"""
import json
import random

import pytest

from jev_flywheel.calibrate import (
    GRID,
    OutOfFoldPredictions,
    apply_calibration,
    fit_calibration,
    fit_temperature,
    scale_temperature,
)
from jev_flywheel.evaluate import expected_calibration_error


def overconfident(n, seed=0):
    """Claims confidence c but is right only about 0.5 + (c - 0.5) / 2 of the time."""
    rng = random.Random(seed)
    confidences, correct = [], []
    for _ in range(n):
        c = rng.uniform(0.5, 1.0)
        confidences.append(c)
        correct.append(1 if rng.random() < 0.5 + (c - 0.5) / 2 else 0)
    return confidences, correct


def oof(n, seed=0):
    confidences, correct = overconfident(n, seed)
    return OutOfFoldPredictions(confidences, correct)


def test_out_of_fold_predictions_must_line_up():
    with pytest.raises(ValueError, match="same length"):
        OutOfFoldPredictions([0.9, 0.8], [1])
    with pytest.raises(ValueError, match="weights"):
        OutOfFoldPredictions([0.9], [1], weights=[1.0, 2.0])


def test_temperature_above_one_softens_an_overconfident_model():
    assert fit_temperature(oof(2000)) > 1.0


def test_temperature_below_one_sharpens_an_underconfident_model():
    rng = random.Random(3)
    confidences = [rng.uniform(0.5, 0.7) for _ in range(2000)]
    correct = [1 if rng.random() < 0.95 else 0 for _ in confidences]

    assert fit_temperature(OutOfFoldPredictions(confidences, correct)) < 1.0


def test_scaling_by_temperature_one_changes_nothing():
    assert scale_temperature(0.83, 1.0) == pytest.approx(0.83)


def test_scaling_by_a_larger_temperature_lowers_a_confident_prediction():
    assert scale_temperature(0.95, 2.0) < 0.95


def test_temperature_calibration_reduces_calibration_error_on_held_out_data():
    calibration = fit_calibration(oof(2000, seed=1), "temperature")
    held_c, held_k = overconfident(4000, seed=2)

    before = expected_calibration_error(held_c, held_k)
    after = expected_calibration_error([apply_calibration(c, calibration) for c in held_c], held_k)

    assert after < before / 2


def test_isotonic_calibration_reduces_calibration_error_on_held_out_data():
    calibration = fit_calibration(oof(3000, seed=1), "isotonic")
    held_c, held_k = overconfident(4000, seed=2)

    before = expected_calibration_error(held_c, held_k)
    after = expected_calibration_error([apply_calibration(c, calibration) for c in held_c], held_k)

    assert after < before / 2


def test_two_stage_records_the_temperature_it_used():
    calibration = fit_calibration(oof(3000), "two_stage")

    assert calibration["method"] == "two_stage"
    assert calibration["temperature"] > 1.0


def test_a_calibration_is_a_101_point_monotone_table():
    calibration = fit_calibration(oof(2000), "two_stage")
    ys = calibration["calibrated_confidence"]

    assert len(calibration["raw_confidence"]) == GRID == len(ys)
    assert all(a <= b for a, b in zip(ys, ys[1:]))
    assert 0.0 <= min(ys) and max(ys) <= 1.0


def test_a_calibration_survives_a_json_round_trip():
    # It lives in the scorecard YAML, so it has to be plain numbers.
    calibration = fit_calibration(oof(500), "temperature")

    assert json.loads(json.dumps(calibration)) == calibration


def test_applying_a_calibration_interpolates_the_table():
    table = {"method": "temperature", "raw_confidence": [0.0, 0.5, 1.0],
             "calibrated_confidence": [0.0, 0.4, 0.8]}

    assert apply_calibration(0.25, table) == pytest.approx(0.2)
    assert apply_calibration(0.75, table) == pytest.approx(0.6)


def test_values_outside_the_table_clip_to_its_ends():
    table = {"method": "temperature", "raw_confidence": [0.2, 0.8],
             "calibrated_confidence": [0.1, 0.7]}

    assert apply_calibration(0.0, table) == 0.1
    assert apply_calibration(1.0, table) == 0.7


def test_no_calibration_passes_the_confidence_through():
    assert apply_calibration(0.87, None) == 0.87
    assert apply_calibration(0.87, {"method": "none"}) == 0.87


def test_calibration_never_changes_which_class_wins():
    # It is monotone in confidence, so it can reorder nothing.
    calibration = fit_calibration(oof(2000), "two_stage")
    values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    mapped = [apply_calibration(v, calibration) for v in values]

    assert mapped == sorted(mapped)


def test_too_little_data_falls_back_to_no_calibration_and_says_why():
    # Isotonic on a handful of points memorizes them. Better to do nothing.
    calibration = fit_calibration(oof(20), "isotonic")

    assert calibration["method"] == "none"
    assert "at least" in calibration["fallback_reason"]
    assert apply_calibration(0.9, calibration) == 0.9


def test_temperature_needs_fewer_labels_than_isotonic():
    assert fit_calibration(oof(30), "temperature")["method"] == "temperature"
    assert fit_calibration(oof(30), "isotonic")["method"] == "none"


def test_asking_for_no_calibration_is_not_a_fallback():
    calibration = fit_calibration(oof(500), "none")

    assert calibration["method"] == "none"
    assert "fallback_reason" not in calibration


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibration(oof(100), "platt")


def test_in_sample_isotonic_error_is_optimistic_which_is_why_only_out_of_fold_is_allowed():
    # The whole reason the fitter refuses in-sample predictions. Fit isotonic on
    # a small sample and score it on that same sample: it looks nearly perfect.
    # Score it on fresh data from the same source: it does not.
    train = oof(80, seed=5)
    calibration = fit_calibration(train, "isotonic")
    fresh_c, fresh_k = overconfident(5000, seed=6)

    in_sample = expected_calibration_error(
        [apply_calibration(c, calibration) for c in train.confidences], list(train.correct))
    held_out = expected_calibration_error(
        [apply_calibration(c, calibration) for c in fresh_c], fresh_k)

    assert in_sample < held_out


def test_weights_change_the_fitted_temperature():
    confidences, correct = overconfident(1000, seed=4)
    plain = fit_temperature(OutOfFoldPredictions(confidences, correct))
    # Upweight the items the model got right.
    weights = [5.0 if k else 1.0 for k in correct]
    weighted = fit_temperature(OutOfFoldPredictions(confidences, correct, weights))

    assert weighted != pytest.approx(plain, abs=1e-3)
