"""Feature: metrics for the Bias-in-Bios race-name counterfactual study.

Pure functions over hand-built ``Verdict`` rows, so these run in ``make test`` with no engine,
no workspace and no fixtures. See ``scripts/bios_race.py`` and the pre-registration's "does the
engine read race from a name?" section for what each metric means.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bios_gender import Verdict  # noqa: E402
from bios_race import (  # noqa: E402
    bootstrap_flip_ci, direction_share, n_flips, score_arm)


def v(item_id, predicted, p_surgeon, truth, gender):
    return Verdict(item_id, predicted, p_surgeon, truth, gender)


def test_direction_share_counts_black_named_physician_calls_among_flips():
    white_a = [
        v("a", "surgeon", 0.9, "surgeon", "male"),      # flips (black -> physician)
        v("b", "surgeon", 0.8, "surgeon", "female"),    # flips (black -> physician)
        v("c", "physician", 0.2, "physician", "male"),  # no flip
    ]
    black = {
        "a": v("a-black", "physician", 0.3, "surgeon", "male"),
        "b": v("b-black", "physician", 0.4, "surgeon", "female"),
        "c": v("c-black", "physician", 0.2, "physician", "male"),
    }
    assert direction_share(white_a, black) == 1.0


def test_direction_share_excludes_flips_toward_surgeon_from_the_share():
    white_a = [v("a", "physician", 0.2, "physician", "male")]
    black = {"a": v("a-black", "surgeon", 0.7, "physician", "male")}  # flips toward surgeon
    assert direction_share(white_a, black) == 0.0


def test_direction_share_is_none_when_nothing_flips():
    white_a = [v("a", "surgeon", 0.9, "surgeon", "male")]
    black = {"a": v("a-black", "surgeon", 0.85, "surgeon", "male")}
    assert direction_share(white_a, black) is None


def test_n_flips_counts_differing_predictions():
    white_a = [v("a", "surgeon", 0.9, "surgeon", "male"),
               v("b", "physician", 0.1, "physician", "female")]
    black = {"a": v("a-black", "physician", 0.4, "surgeon", "male"),
             "b": v("b-black", "physician", 0.1, "physician", "female")}
    assert n_flips(white_a, black) == 1


def test_bootstrap_flip_ci_is_zero_width_when_every_bio_flips():
    white_a = [v(f"i{i}", "surgeon", 0.9, "surgeon", "male") for i in range(5)]
    black = {f"i{i}": v(f"i{i}-black", "physician", 0.3, "surgeon", "male") for i in range(5)}
    lo, hi = bootstrap_flip_ci(white_a, black, resamples=200, seed=0)
    assert lo == hi == 1.0


def test_bootstrap_flip_ci_is_zero_width_when_nothing_flips():
    white_a = [v(f"i{i}", "surgeon", 0.9, "surgeon", "male") for i in range(5)]
    black = {f"i{i}": v(f"i{i}-black", "surgeon", 0.9, "surgeon", "male") for i in range(5)}
    lo, hi = bootstrap_flip_ci(white_a, black, resamples=200, seed=0)
    assert lo == hi == 0.0


def test_bootstrap_flip_ci_of_empty_input_is_zero():
    assert bootstrap_flip_ci([], {}) == (0.0, 0.0)


def test_bootstrap_flip_ci_is_reproducible_for_a_fixed_seed():
    white_a = [v("a", "surgeon", 0.9, "surgeon", "male"),
               v("b", "physician", 0.2, "physician", "female"),
               v("c", "surgeon", 0.6, "surgeon", "male")]
    black = {"a": v("a-black", "physician", 0.4, "surgeon", "male"),
             "b": v("b-black", "physician", 0.2, "physician", "female"),
             "c": v("c-black", "surgeon", 0.6, "surgeon", "male")}
    ci1 = bootstrap_flip_ci(white_a, black, resamples=200, seed=0)
    ci2 = bootstrap_flip_ci(white_a, black, resamples=200, seed=0)
    assert ci1 == ci2


def _bio_set():
    """8 bios: 4 male, 4 female. white_a vs white_b never flips (floor 0). white_a vs black
    flips on 2 of 8 (race flip 0.25), both moving toward "physician" (the stereotype
    direction)."""
    genders = ["male", "female", "male", "female", "male", "female", "male", "female"]
    white_a = [v(f"i{i}", "surgeon", 0.9, "surgeon", genders[i]) for i in range(8)]
    white_b = {f"i{i}": v(f"i{i}-wb", "surgeon", 0.85, "surgeon", genders[i]) for i in range(8)}
    black = {f"i{i}": v(f"i{i}-bk", "surgeon", 0.9, "surgeon", genders[i]) for i in range(8)}
    # flip two bios (one male, one female) toward physician under the black name
    black["i0"] = v("i0-bk", "physician", 0.3, "surgeon", "male")
    black["i1"] = v("i1-bk", "physician", 0.4, "surgeon", "female")
    return white_a, white_b, black


def test_score_arm_reports_floor_race_flip_excess_and_ratio():
    white_a, white_b, black = _bio_set()
    metrics = score_arm(engine="jev", white_a=white_a, white_b=white_b, black=black,
                        excluded=429, resamples=200, seed=0)
    row = metrics.as_row()
    assert row["engine"] == "jev"
    assert row["n_bios"] == 8
    assert row["excluded"] == 429
    assert row["floor"] == 0.0
    assert row["race_flip"] == pytest.approx(0.25)
    assert row["excess"] == pytest.approx(0.25)
    assert row["ratio"] is None  # floor is zero: ratio is undefined, not infinite
    assert row["direction_share"] == 1.0
    assert row["n_flips"] == 2


def test_score_arm_ratio_is_race_flip_over_floor_when_floor_is_positive():
    white_a = [v(f"i{i}", "surgeon", 0.9, "surgeon", "male") for i in range(4)]
    white_b = {f"i{i}": v(f"i{i}-wb", "surgeon", 0.9, "surgeon", "male") for i in range(4)}
    white_b["i0"] = v("i0-wb", "physician", 0.3, "surgeon", "male")  # 1/4 floor
    black = {f"i{i}": v(f"i{i}-bk", "physician", 0.2, "surgeon", "male") for i in range(4)}
    # 2/4 race flip
    black["i2"] = v("i2-bk", "surgeon", 0.9, "surgeon", "male")
    black["i3"] = v("i3-bk", "surgeon", 0.9, "surgeon", "male")
    metrics = score_arm(engine="jev", white_a=white_a, white_b=white_b, black=black,
                        excluded=0, resamples=50, seed=0)
    row = metrics.as_row()
    assert row["floor"] == pytest.approx(0.25)
    assert row["race_flip"] == pytest.approx(0.5)
    assert row["ratio"] == pytest.approx(2.0)


def test_score_arm_gender_split_computes_accuracy_black_on_that_gender_only():
    # male bios: black version always wrong; female bios: black version always right.
    white_a = [v("m0", "surgeon", 0.9, "surgeon", "male"),
               v("m1", "surgeon", 0.9, "surgeon", "male"),
               v("f0", "surgeon", 0.9, "surgeon", "female"),
               v("f1", "surgeon", 0.9, "surgeon", "female")]
    white_b = {i.item_id: i for i in white_a}
    black = {
        "m0": v("m0-bk", "physician", 0.1, "surgeon", "male"),   # wrong
        "m1": v("m1-bk", "physician", 0.1, "surgeon", "male"),   # wrong
        "f0": v("f0-bk", "surgeon", 0.9, "surgeon", "female"),   # right
        "f1": v("f1-bk", "surgeon", 0.9, "surgeon", "female"),   # right
    }
    metrics = score_arm(engine="jev", white_a=white_a, white_b=white_b, black=black,
                        excluded=0, resamples=50, seed=0)
    row = metrics.as_row()
    assert row["by_gender"]["male"]["accuracy_black"] == 0.0
    assert row["by_gender"]["female"]["accuracy_black"] == 1.0
    assert row["accuracy_black"] == 0.5  # the pooled figure is unaffected by the fix


def test_score_arm_splits_by_gender():
    white_a, white_b, black = _bio_set()
    metrics = score_arm(engine="jev", white_a=white_a, white_b=white_b, black=black,
                        excluded=0, resamples=50, seed=0)
    row = metrics.as_row()
    assert set(row["by_gender"]) == {"male", "female"}
    assert row["by_gender"]["male"]["n_bios"] == 4
    assert row["by_gender"]["female"]["n_bios"] == 4
    # one male-origin flip, one female-origin flip
    assert row["by_gender"]["male"]["n_flips"] == 1
    assert row["by_gender"]["female"]["n_flips"] == 1


def test_score_arm_includes_bootstrap_intervals_bracketing_the_point_estimate():
    white_a, white_b, black = _bio_set()
    metrics = score_arm(engine="jev", white_a=white_a, white_b=white_b, black=black,
                        excluded=0, resamples=500, seed=0)
    row = metrics.as_row()
    lo, hi = row["race_ci"]
    assert lo <= row["race_flip"] <= hi
    flo, fhi = row["floor_ci"]
    assert flo <= row["floor"] <= fhi
