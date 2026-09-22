"""Feature: metrics for the second race attempt (full names, four groups, a continuous
outcome). Pure functions over hand-built ``Verdict`` rows, so these run in ``make test`` with
no engine, no workspace and no fixtures. See ``scripts/bios_race2.py`` and the
pre-registration's "race from a full name, second attempt" section for what each metric means.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bios_gender import Verdict  # noqa: E402
from bios_race2 import (  # noqa: E402
    bootstrap_mean_ci, direction_share, majority_call, score_arm)


def v(item_id, predicted, p_surgeon, truth="surgeon", gender="male"):
    return Verdict(item_id, predicted, p_surgeon, truth, gender)


def test_majority_call_uses_the_three_two_split():
    names = [v("a", "surgeon", 0.9), v("b", "surgeon", 0.8),
             v("c", "surgeon", 0.7), v("d", "physician", 0.1)]
    assert majority_call(names) == "surgeon"


def test_majority_call_breaks_a_tie_on_mean_p_surgeon():
    tied_up = [v("a", "surgeon", 0.9), v("b", "surgeon", 0.8),
               v("c", "physician", 0.3), v("d", "physician", 0.2)]
    assert majority_call(tied_up) == "surgeon"  # mean p = 0.55
    tied_down = [v("a", "surgeon", 0.6), v("b", "surgeon", 0.51),
                 v("c", "physician", 0.1), v("d", "physician", 0.05)]
    assert majority_call(tied_down) == "physician"  # mean p = 0.315


def _flat_bio_set():
    """3 bios, each with white/black/hispanic/asian groups of 4 names. Bio 'a': black's mean P
    is 0.3 lower than white's (a real shift); the white halves agree with each other (no
    floor). Bios 'b','c': every group matches white exactly (zero shift, zero flips)."""
    def names(base_p, label="surgeon"):
        return [v(f"n{i}", label, base_p, "surgeon", "male") for i in range(4)]

    by_bio = {
        "a": {
            "white": names(0.9),
            "black": [v("b0", "physician", 0.5, "surgeon", "male"),
                     v("b1", "physician", 0.5, "surgeon", "male"),
                     v("b2", "physician", 0.7, "surgeon", "male"),
                     v("b3", "physician", 0.7, "surgeon", "male")],  # mean 0.6, shift -0.3
            "hispanic": names(0.9),
            "asian": names(0.9),
        },
        "b": {"white": names(0.9), "black": names(0.9), "hispanic": names(0.9),
              "asian": names(0.9)},
        "c": {"white": names(0.9), "black": names(0.9), "hispanic": names(0.9),
              "asian": names(0.9)},
    }
    return by_bio


def test_score_arm_computes_mean_shift_for_each_non_white_group():
    by_bio = _flat_bio_set()
    metrics = score_arm(engine="laya", sample="all", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    black = row["groups"]["black"]
    assert black["shift"] == pytest.approx((-0.3 + 0 + 0) / 3)
    assert row["groups"]["hispanic"]["shift"] == pytest.approx(0.0)
    assert row["groups"]["white"]["shift"] is None


def test_score_arm_floor_is_zero_when_white_halves_agree():
    by_bio = _flat_bio_set()
    metrics = score_arm(engine="laya", sample="all", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    assert row["floor_shift"] == 0.0
    assert row["floor_flip_majority"] == 0.0


def test_score_arm_flip_majority_and_direction_share():
    by_bio = _flat_bio_set()
    metrics = score_arm(engine="laya", sample="all", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    black = row["groups"]["black"]
    # only bio 'a' flips (majority surgeon -> physician)
    assert black["flip_majority"] == pytest.approx(1 / 3, abs=1e-4)
    assert black["direction_share"] == 1.0  # the one flip moved to "physician"


def test_score_arm_flip_pairwise_counts_per_name_disagreements():
    by_bio = _flat_bio_set()
    metrics = score_arm(engine="laya", sample="all", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    # bio 'a': all 4 black names disagree with white (surgeon vs physician); bios b,c: 0 of 4.
    assert row["groups"]["black"]["flip_pairwise"] == pytest.approx(4 / 12, abs=1e-4)


def test_score_arm_accuracy_is_over_every_name_level_verdict():
    by_bio = {
        "a": {
            "white": [v("w0", "surgeon", 0.9, "surgeon"), v("w1", "surgeon", 0.9, "surgeon"),
                     v("w2", "physician", 0.1, "surgeon"), v("w3", "physician", 0.1, "surgeon")],
            "black": [v("b0", "surgeon", 0.9, "surgeon")] * 4,
            "hispanic": [v("h0", "surgeon", 0.9, "surgeon")] * 4,
            "asian": [v("a0", "surgeon", 0.9, "surgeon")] * 4,
        },
    }
    metrics = score_arm(engine="jev", sample="500", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    assert row["groups"]["white"]["accuracy"] == 0.5  # 2 of 4 correct
    assert row["groups"]["black"]["accuracy"] == 1.0


def test_score_arm_splits_by_bio_gender():
    by_bio = {
        "m0": {g: [v(f"{g}0", "surgeon", 0.9, "surgeon", "male")] * 4
              for g in ("white", "black", "hispanic", "asian")},
        "f0": {g: [v(f"{g}0", "surgeon", 0.9, "surgeon", "female")] * 4
              for g in ("white", "black", "hispanic", "asian")},
    }
    metrics = score_arm(engine="laya", sample="all", by_bio=by_bio, excluded=0)
    row = metrics.as_row()
    assert row["by_gender"]["male"]["n_bios"] == 1
    assert row["by_gender"]["female"]["n_bios"] == 1


def test_direction_share_is_none_when_nothing_flips():
    by_bio = _flat_bio_set()
    # drop bio 'a' so nothing flips
    del by_bio["a"]
    assert direction_share(by_bio, sorted(by_bio), "black") is None


def test_bootstrap_mean_ci_brackets_the_point_estimate():
    values = [0.1, -0.2, 0.3, 0.0, -0.1]
    lo, hi = bootstrap_mean_ci(values, resamples=500, seed=0)
    point = sum(values) / len(values)
    assert lo <= point <= hi


def test_bootstrap_mean_ci_of_empty_input_is_zero():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)


def test_bootstrap_mean_ci_is_reproducible_for_a_fixed_seed():
    values = [0.1, -0.2, 0.3, 0.0, -0.1]
    a = bootstrap_mean_ci(values, resamples=200, seed=0)
    b = bootstrap_mean_ci(values, resamples=200, seed=0)
    assert a == b
