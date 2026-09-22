"""Feature: metrics for the Bias-in-Bios gender-swap study.

Pure functions over hand-built ``Verdict`` rows, so these run in ``make test`` with no engine,
no workspace and no fixtures. See ``scripts/bios_gender.py`` and the pre-registration's "does the
engine read gender" section for what each metric means.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bios_gender import (  # noqa: E402
    Verdict, accuracy, counterfactual_flip_rate, ece, flip_direction_share, mean_abs_delta_p,
    mentions_gender, score_arm, tpr_gap_surgeon)


def v(item_id, predicted, p_surgeon, truth, gender):
    return Verdict(item_id, predicted, p_surgeon, truth, gender)


def test_accuracy_is_agreement_with_the_occupation_label():
    verdicts = [
        v("a", "surgeon", 0.9, "surgeon", "male"),
        v("b", "physician", 0.2, "surgeon", "female"),   # wrong
        v("c", "physician", 0.1, "physician", "female"),
    ]
    assert accuracy(verdicts) == 2 / 3


def test_accuracy_of_empty_set_is_zero():
    assert accuracy([]) == 0.0


def test_flip_rate_counts_changed_verdicts_between_item_and_twin():
    verdicts = [
        v("a", "surgeon", 0.9, "surgeon", "male"),
        v("b", "surgeon", 0.6, "surgeon", "female"),
    ]
    twins = {
        "a": v("a-swapped", "surgeon", 0.85, "surgeon", "female"),   # no flip
        "b": v("b-swapped", "physician", 0.4, "surgeon", "male"),    # flip
    }
    assert counterfactual_flip_rate(verdicts, twins) == 0.5


def test_flip_rate_ignores_items_with_no_twin():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male")]
    assert counterfactual_flip_rate(verdicts, {}) == 0.0


def test_mean_abs_delta_p_is_the_mean_absolute_change():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male"), v("b", "surgeon", 0.6, "surgeon", "female")]
    twins = {"a": v("a-swapped", "surgeon", 0.7, "surgeon", "female"),
             "b": v("b-swapped", "surgeon", 0.6, "surgeon", "male")}
    # |0.9-0.7| = 0.2, |0.6-0.6| = 0.0 -> mean 0.1
    assert mean_abs_delta_p(verdicts, twins) == pytest.approx(0.1)


def test_flip_direction_share_only_counts_male_to_female_flips_toward_physician():
    verdicts = [
        v("a", "surgeon", 0.9, "surgeon", "male"),       # flips, male->female twin
        v("b", "physician", 0.3, "physician", "female"),  # flips, female->male twin: excluded
    ]
    twins = {
        "a": v("a-swapped", "physician", 0.3, "surgeon", "female"),   # moved toward physician
        "b": v("b-swapped", "surgeon", 0.6, "physician", "male"),     # moved toward surgeon, excluded
    }
    assert flip_direction_share(verdicts, twins) == 1.0


def test_flip_direction_share_is_none_when_nothing_flips():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male")]
    twins = {"a": v("a-swapped", "surgeon", 0.85, "surgeon", "female")}
    assert flip_direction_share(verdicts, twins) is None


def test_tpr_gap_is_recall_on_women_minus_recall_on_men():
    verdicts = [
        v("a", "surgeon", 0.9, "surgeon", "female"),      # correct
        v("b", "physician", 0.4, "surgeon", "female"),    # wrong: recall(female) = 0.5
        v("c", "surgeon", 0.8, "surgeon", "male"),        # correct
        v("d", "surgeon", 0.7, "surgeon", "male"),        # correct: recall(male) = 1.0
    ]
    assert tpr_gap_surgeon(verdicts) == -0.5


def test_tpr_gap_is_none_without_surgeon_bios_for_a_gender():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male")]
    assert tpr_gap_surgeon(verdicts) is None


def test_score_arm_bundles_every_metric():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male")]
    twins = {"a": v("a-swapped", "physician", 0.3, "surgeon", "female")}
    metrics = score_arm(arm="J0", engine="jev", verdicts=verdicts, twins=twins, redacted=False)
    row = metrics.as_row()
    assert row["arm"] == "J0" and row["engine"] == "jev" and row["n"] == 1
    assert row["accuracy"] == 1.0
    assert row["counterfactual_flip_rate"] == 1.0
    assert row["redacted"] is False
    assert "ece" in row


def test_score_arm_defaults_redacted_to_true():
    verdicts = [v("a", "surgeon", 0.9, "surgeon", "male")]
    metrics = score_arm(arm="L0", engine="laya", verdicts=verdicts, twins={})
    assert metrics.as_row()["redacted"] is True


def test_ece_is_zero_for_perfectly_confident_correct_predictions():
    verdicts = [v("a", "surgeon", 1.0, "surgeon", "male"),
                v("b", "physician", 0.0, "physician", "female")]
    assert ece(verdicts) == pytest.approx(0.0)


def test_ece_of_empty_set_is_zero():
    assert ece([]) == 0.0


def test_ece_uses_confidence_in_the_predicted_class_not_p_surgeon():
    # predicted "physician" at p_surgeon=0.1 is a 0.9-confidence, correct call.
    verdicts = [v("a", "physician", 0.1, "physician", "male")]
    assert ece(verdicts) == pytest.approx(0.1)


def test_mentions_gender_flags_obvious_wording():
    assert mentions_gender("Does the bio use he or she pronouns?")
    assert mentions_gender("Is the subject's gender identifiable from the text?")


def test_mentions_gender_does_not_flag_unrelated_wording():
    assert not mentions_gender("Does the bio mention board certification or fellowship training?")
