"""Feature: metrics for the Bias-in-Bios age-insertion counterfactual study.

Pure functions over hand-built ``Verdict`` rows, so these run in ``make test`` with no engine,
no workspace and no fixtures. See ``scripts/bios_age.py`` and the pre-registration's "does the
engine read age?" section for what each metric means.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bios_gender import Verdict  # noqa: E402
from bios_age import (  # noqa: E402
    bootstrap_ci, direction_older_surgeon_share, n_flips, score_arm, signed_mean_shift,
    _flip_stat, _shift_stat)


def v(item_id, predicted, p_surgeon, truth, gender):
    return Verdict(item_id, predicted, p_surgeon, truth, gender)


def test_signed_mean_shift_is_positive_when_older_reads_more_surgeon():
    v34 = [v("a", "physician", 0.3, "surgeon", "male"),
           v("b", "surgeon", 0.6, "surgeon", "female")]
    v61 = {"a": v("a-61", "surgeon", 0.5, "surgeon", "male"),
           "b": v("b-61", "surgeon", 0.8, "surgeon", "female")}
    # deltas: 0.2, 0.2 -> mean 0.2
    assert signed_mean_shift(v34, v61) == pytest.approx(0.2)


def test_signed_mean_shift_is_zero_with_no_pairs():
    assert signed_mean_shift([], {}) == 0.0


def test_direction_share_counts_older_named_surgeon_calls_among_flips():
    v34 = [
        v("a", "physician", 0.3, "surgeon", "male"),    # flips (34->61 physician->surgeon)
        v("b", "physician", 0.2, "physician", "female"),  # flips
        v("c", "surgeon", 0.9, "surgeon", "male"),        # no flip
    ]
    v61 = {
        "a": v("a-61", "surgeon", 0.7, "surgeon", "male"),
        "b": v("b-61", "surgeon", 0.6, "physician", "female"),
        "c": v("c-61", "surgeon", 0.9, "surgeon", "male"),
    }
    assert direction_older_surgeon_share(v34, v61) == 1.0


def test_direction_share_excludes_flips_toward_physician_from_the_numerator():
    v34 = [v("a", "surgeon", 0.7, "surgeon", "male")]
    v61 = {"a": v("a-61", "physician", 0.3, "surgeon", "male")}  # flips toward physician
    assert direction_older_surgeon_share(v34, v61) == 0.0


def test_direction_share_is_none_when_nothing_flips():
    v34 = [v("a", "surgeon", 0.9, "surgeon", "male")]
    v61 = {"a": v("a-61", "surgeon", 0.85, "surgeon", "male")}
    assert direction_older_surgeon_share(v34, v61) is None


def test_n_flips_counts_differing_predictions():
    v34 = [v("a", "surgeon", 0.9, "surgeon", "male"),
           v("b", "physician", 0.1, "physician", "female")]
    v61 = {"a": v("a-61", "physician", 0.4, "surgeon", "male"),
           "b": v("b-61", "physician", 0.1, "physician", "female")}
    assert n_flips(v34, v61) == 1


def test_bootstrap_ci_flip_stat_is_zero_width_when_every_bio_flips():
    v34 = [v(f"i{i}", "surgeon", 0.9, "surgeon", "male") for i in range(5)]
    v61 = {f"i{i}": v(f"i{i}-61", "physician", 0.3, "surgeon", "male") for i in range(5)}
    lo, hi = bootstrap_ci(v34, v61, _flip_stat, resamples=200, seed=0)
    assert lo == hi == 1.0


def test_bootstrap_ci_flip_stat_is_zero_width_when_nothing_flips():
    v34 = [v(f"i{i}", "surgeon", 0.9, "surgeon", "male") for i in range(5)]
    v61 = {f"i{i}": v(f"i{i}-61", "surgeon", 0.9, "surgeon", "male") for i in range(5)}
    lo, hi = bootstrap_ci(v34, v61, _flip_stat, resamples=200, seed=0)
    assert lo == hi == 0.0


def test_bootstrap_ci_of_empty_input_is_zero():
    assert bootstrap_ci([], {}, _flip_stat) == (0.0, 0.0)


def test_bootstrap_ci_is_reproducible_for_a_fixed_seed():
    v34 = [v("a", "surgeon", 0.9, "surgeon", "male"),
           v("b", "physician", 0.2, "physician", "female"),
           v("c", "surgeon", 0.6, "surgeon", "male")]
    v61 = {"a": v("a-61", "physician", 0.4, "surgeon", "male"),
           "b": v("b-61", "physician", 0.2, "physician", "female"),
           "c": v("c-61", "surgeon", 0.6, "surgeon", "male")}
    ci1 = bootstrap_ci(v34, v61, _shift_stat, resamples=200, seed=0)
    ci2 = bootstrap_ci(v34, v61, _shift_stat, resamples=200, seed=0)
    assert ci1 == ci2


def _bio_set():
    """8 bios: 4 male, 4 female. 34 vs 35 and 61 vs 62 never flip (floors 0). 34 vs 61 flips on
    2 of 8 (age flip 0.25), both moving toward "surgeon" (the seniority-association
    direction)."""
    genders = ["male", "female", "male", "female", "male", "female", "male", "female"]
    v34 = [v(f"i{i}", "physician", 0.3, "surgeon", genders[i]) for i in range(8)]
    v35 = {f"i{i}": v(f"i{i}-35", "physician", 0.32, "surgeon", genders[i]) for i in range(8)}
    v61 = {f"i{i}": v(f"i{i}-61", "physician", 0.3, "surgeon", genders[i]) for i in range(8)}
    v62 = {f"i{i}": v(f"i{i}-62", "physician", 0.31, "surgeon", genders[i]) for i in range(8)}
    # flip two bios (one male, one female) toward surgeon under the 61 version
    v61["i0"] = v("i0-61", "surgeon", 0.7, "surgeon", "male")
    v61["i1"] = v("i1-61", "surgeon", 0.8, "surgeon", "female")
    return v34, v35, v61, v62


def test_score_arm_reports_age_flip_and_floors():
    v34, v35, v61, v62 = _bio_set()
    metrics = score_arm(engine="jev", v34=v34, v35=v35, v61=v61, v62=v62, excluded=769,
                        resamples=200, seed=0)
    row = metrics.as_row()
    assert row["engine"] == "jev"
    assert row["n_bios"] == 8
    assert row["excluded"] == 769
    assert row["floor_35_flip"] == 0.0
    assert row["floor_62_flip"] == 0.0
    assert row["age_flip"] == pytest.approx(0.25)
    assert row["direction_share"] == 1.0
    assert row["n_flips_age"] == 2
    assert row["age_shift"] > 0  # both flips moved toward surgeon


def test_score_arm_splits_by_gender():
    v34, v35, v61, v62 = _bio_set()
    metrics = score_arm(engine="jev", v34=v34, v35=v35, v61=v61, v62=v62, excluded=0,
                        resamples=50, seed=0)
    row = metrics.as_row()
    assert set(row["by_gender"]) == {"male", "female"}
    assert row["by_gender"]["male"]["n_bios"] == 4
    assert row["by_gender"]["female"]["n_bios"] == 4
    assert row["by_gender"]["male"]["n_flips_age"] == 1
    assert row["by_gender"]["female"]["n_flips_age"] == 1


def test_score_arm_includes_bootstrap_intervals_bracketing_the_point_estimate():
    v34, v35, v61, v62 = _bio_set()
    metrics = score_arm(engine="jev", v34=v34, v35=v35, v61=v61, v62=v62, excluded=0,
                        resamples=500, seed=0)
    row = metrics.as_row()
    lo, hi = row["age_flip_ci"]
    assert lo <= row["age_flip"] <= hi
    lo, hi = row["age_shift_ci"]
    assert lo <= row["age_shift"] <= hi
    lo, hi = row["floor_35_flip_ci"]
    assert lo <= row["floor_35_flip"] <= hi
    lo, hi = row["floor_62_flip_ci"]
    assert lo <= row["floor_62_flip"] <= hi


def test_score_arm_accuracy_per_version():
    v34, v35, v61, v62 = _bio_set()
    metrics = score_arm(engine="jev", v34=v34, v35=v35, v61=v61, v62=v62, excluded=0,
                        resamples=50, seed=0)
    row = metrics.as_row()
    # v34/v35/v62 are all "physician" against a "surgeon" truth -> accuracy 0
    assert row["accuracy_34"] == 0.0
    assert row["accuracy_35"] == 0.0
    assert row["accuracy_62"] == 0.0
    # v61: 6 of 8 still "physician" (wrong), 2 flipped to "surgeon" (right) -> 0.25
    assert row["accuracy_61"] == pytest.approx(0.25)
