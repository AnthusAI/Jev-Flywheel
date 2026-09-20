"""Feature: deciding when a refit or a rethink is worth it."""
import pytest

from jev_flywheel.steering import (
    POLICY_NAME,
    SteeringPolicy,
    SteeringState,
    evaluate,
)


def state(**overrides):
    """A state where every condition of both triggers is met, then overridden."""
    base = dict(
        n_labeled=80, n_effective=80.0, labels_since_fit=6, labels_since_rethink=20,
        commented_mismatches_since_rethink=6,
        # Fast early progress, then flat: 0.470 -> 0.465 over the last three refits.
        fit_log_losses=[0.60, 0.50, 0.470, 0.469, 0.466, 0.465], oof_accuracy=0.85)
    base.update(overrides)
    return SteeringState(**base)


def test_everything_met_fires_both_triggers():
    triggers = evaluate(state())

    assert triggers["refit"].fire
    assert triggers["rethink"].fire


def test_a_refit_waits_until_there_is_enough_evidence_to_fit_anything():
    trigger = evaluate(state(n_effective=12.0))["refit"]

    assert not trigger.fire
    assert trigger.unmet[0].name == "enough evidence to fit"


def test_a_refit_waits_for_a_few_new_labels_rather_than_running_on_every_one():
    assert not evaluate(state(labels_since_fit=2))["refit"].fire
    assert evaluate(state(labels_since_fit=5))["refit"].fire


def test_a_rethink_needs_a_head_that_can_be_fit_because_a_candidate_must_be_comparable():
    # Proposing elements it cannot evaluate would mean promoting blind.
    trigger = evaluate(state(n_effective=20.0))["rethink"]

    assert not trigger.fire
    assert trigger.unmet[0].name == "a candidate can be evaluated"


def test_a_rethink_needs_the_human_to_have_explained_some_disagreements():
    trigger = evaluate(state(commented_mismatches_since_rethink=2))["rethink"]

    assert not trigger.fire
    assert [c.name for c in trigger.unmet] == ["mismatches the human explained"]


def test_a_rethink_respects_a_cooldown_so_it_does_not_interrupt_repeatedly():
    trigger = evaluate(state(labels_since_rethink=4))["rethink"]

    assert [c.name for c in trigger.unmet] == ["cooldown since the last rethink"]


def test_a_rethink_waits_for_the_plateau_that_says_features_are_the_bottleneck():
    # Still improving quickly: more labels will help, better questions are premature.
    improving = [0.90, 0.80, 0.70, 0.60, 0.50]

    trigger = evaluate(state(fit_log_losses=improving))["rethink"]

    assert not trigger.fire
    assert [c.name for c in trigger.unmet] == ["progress has plateaued"]


def test_too_few_refits_cannot_show_a_plateau():
    trigger = evaluate(state(fit_log_losses=[0.6, 0.6]))["rethink"]

    unmet = trigger.unmet[0]
    assert unmet.name == "progress has plateaued"
    assert "too few" in unmet.have


def test_a_rethink_is_pointless_when_the_head_is_already_nearly_perfect():
    assert not evaluate(state(oof_accuracy=0.995))["rethink"].fire


def test_an_unknown_accuracy_blocks_a_rethink_rather_than_assuming_headroom():
    trigger = evaluate(state(oof_accuracy=None))["rethink"]

    assert [c.name for c in trigger.unmet] == ["room left to improve"]
    assert trigger.unmet[0].have == "unknown"


def test_a_summary_names_the_first_unmet_condition_and_how_far_off_it_is():
    summary = evaluate(state(commented_mismatches_since_rethink=2))["rethink"].summary

    assert "not yet" in summary
    assert "mismatches the human explained" in summary
    assert "have 2" in summary and "need 5" in summary


def test_a_firing_trigger_says_so():
    assert "worth doing now" in evaluate(state())["refit"].summary


def test_a_fresh_workspace_fires_nothing():
    triggers = evaluate(SteeringState())

    assert not triggers["refit"].fire and not triggers["rethink"].fire


def test_thresholds_come_from_the_policy():
    strict = SteeringPolicy(rethink_min_commented_mismatches=20)

    assert not evaluate(state(), strict)["rethink"].fire
    assert evaluate(state())["rethink"].fire


def test_the_policy_is_frozen_and_named_so_a_recorded_decision_stays_interpretable():
    assert SteeringPolicy().name == POLICY_NAME == "steering-v1"
    with pytest.raises(Exception):
        SteeringPolicy().refit_every = 1
