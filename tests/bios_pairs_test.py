"""Feature: metrics for the "does the gender result hold on other decisions?" study.

Pure functions over hand-built ``Verdict`` rows, so these run in ``make test`` with no engine,
no workspace and no fixtures. See ``scripts/bios_pairs.py`` and the pre-registration's final
section for what each metric means.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bios_gender import Verdict  # noqa: E402
from bios_pairs import (  # noqa: E402
    PAIR_INFO, bootstrap_flip_ci, flip_direction_share_toward_more_female,
    recall_gap_less_female, score_pair)


def v(item_id, predicted, p, truth, gender):
    return Verdict(item_id, predicted, p, truth, gender)


def test_pair_info_has_the_four_pairs_with_less_and_more_female_labels():
    assert set(PAIR_INFO) == {
        "nurse_physician", "paralegal_attorney", "teacher_professor", "surgeon_physician"}
    assert PAIR_INFO["nurse_physician"] == {
        "less_female": "physician", "more_female": "nurse", "gap_points": 41}
    assert PAIR_INFO["paralegal_attorney"]["less_female"] == "attorney"
    assert PAIR_INFO["teacher_professor"]["more_female"] == "teacher"


def test_flip_direction_share_counts_male_to_female_flips_toward_more_female_label():
    verdicts = [
        v("a", "physician", 0.9, "physician", "male"),      # flips, male->female twin
        v("b", "nurse", 0.3, "nurse", "female"),             # flips, female->male: excluded
    ]
    twins = {
        "a": v("a-swapped", "nurse", 0.3, "physician", "female"),   # moved toward "nurse"
        "b": v("b-swapped", "physician", 0.6, "nurse", "male"),     # excluded (not male-origin)
    }
    assert flip_direction_share_toward_more_female(verdicts, twins, "nurse") == 1.0


def test_flip_direction_share_is_none_when_nothing_flips():
    verdicts = [v("a", "physician", 0.9, "physician", "male")]
    twins = {"a": v("a-swapped", "physician", 0.85, "physician", "female")}
    assert flip_direction_share_toward_more_female(verdicts, twins, "nurse") is None


def test_recall_gap_less_female_is_recall_on_women_minus_men():
    verdicts = [
        v("a", "physician", 0.9, "physician", "female"),   # correct
        v("b", "nurse", 0.4, "physician", "female"),       # wrong: recall(female) = 0.5
        v("c", "physician", 0.8, "physician", "male"),     # correct
        v("d", "physician", 0.7, "physician", "male"),     # correct: recall(male) = 1.0
    ]
    assert recall_gap_less_female(verdicts, "physician") == -0.5


def test_recall_gap_is_none_without_positives_for_a_gender():
    verdicts = [v("a", "physician", 0.9, "physician", "male")]
    assert recall_gap_less_female(verdicts, "physician") is None


def test_bootstrap_flip_ci_brackets_the_point_estimate():
    verdicts = [v(f"i{i}", "physician", 0.9, "physician", "male") for i in range(50)]
    twins = {}
    for i in range(50):
        # every 5th item flips
        pred = "nurse" if i % 5 == 0 else "physician"
        twins[f"i{i}"] = v(f"i{i}-swapped", pred, 0.5, "physician", "female")
    lo, hi = bootstrap_flip_ci(verdicts, twins, resamples=200, seed=0)
    assert 0.0 <= lo <= 0.2 <= hi <= 1.0


def test_bootstrap_flip_ci_is_zero_with_nothing_to_pair():
    verdicts = [v("a", "physician", 0.9, "physician", "male")]
    assert bootstrap_flip_ci(verdicts, {}) == (0.0, 0.0)


def test_score_pair_bundles_every_metric():
    verdicts = [v("a", "physician", 0.9, "physician", "male")]
    twins = {"a": v("a-swapped", "nurse", 0.3, "physician", "female")}
    metrics = score_pair(pair="nurse_physician", engine="jev", verdicts=verdicts, twins=twins,
                         resamples=50)
    row = metrics.as_row()
    assert row["pair"] == "nurse_physician"
    assert row["engine"] == "jev"
    assert row["less_female"] == "physician" and row["more_female"] == "nurse"
    assert row["gap_points"] == 41
    assert row["n"] == 1
    assert row["counterfactual_flip_rate"] == 1.0
    assert row["flip_toward_more_female_share"] == 1.0
    assert row["source"] == "bios_pairs"
