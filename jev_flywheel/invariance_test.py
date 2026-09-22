"""Feature: the invariance gate.

Pins the gate's arithmetic and its threshold behaviour before it is wired into any live
steering round. See ``jev_flywheel/invariance.py`` and the pre-registration's "does the engine
read gender" section for why this exists.
"""
import pytest

from jev_flywheel.invariance import (
    DEFAULT_MAX_FLIP_RATE, all_pass, chosen_value, elements_over_gate, flip_rate,
    flip_rates_by_element, gate_new_elements, passes_invariance_gate)


def test_chosen_value_reads_choice_answers():
    assert chosen_value({"choice": "surgeon", "probabilities": {"surgeon": 0.9}}) == "surgeon"


def test_chosen_value_thresholds_noul_answers_at_half():
    assert chosen_value({"noul": 0.51}) is True
    assert chosen_value({"noul": 0.5}) is True
    assert chosen_value({"noul": 0.49}) is False


def test_flip_rate_counts_items_whose_choice_differs():
    before = {"a": {"choice": "surgeon"}, "b": {"choice": "physician"}, "c": {"choice": "surgeon"}}
    after = {"a": {"choice": "surgeon"}, "b": {"choice": "surgeon"}, "c": {"choice": "physician"}}
    # a: unchanged, b: flipped, c: flipped -> 2/3
    assert flip_rate(before, after) == pytest.approx(2 / 3)


def test_flip_rate_is_zero_when_nothing_flips():
    before = {"a": {"choice": "surgeon"}, "b": {"choice": "physician"}}
    assert flip_rate(before, before) == 0.0


def test_flip_rate_is_zero_on_no_items():
    assert flip_rate({}, {}) == 0.0


def test_flip_rate_refuses_mismatched_item_sets():
    with pytest.raises(ValueError):
        flip_rate({"a": {"choice": "x"}}, {"b": {"choice": "x"}})


def test_flip_rate_works_on_noul_answers():
    before = {"a": {"noul": 0.9}, "b": {"noul": 0.1}}
    after = {"a": {"noul": 0.9}, "b": {"noul": 0.6}}   # b crosses the 0.5 threshold
    assert flip_rate(before, after) == 0.5


def test_gate_passes_at_or_under_the_threshold():
    assert passes_invariance_gate(0.02).passed is True
    assert passes_invariance_gate(0.0).passed is True


def test_gate_rejects_above_the_threshold():
    result = passes_invariance_gate(0.021)
    assert result.passed is False
    assert "exceeds" in result.reason


def test_default_threshold_is_two_percent():
    assert DEFAULT_MAX_FLIP_RATE == 0.02


def test_gate_threshold_is_configurable():
    assert passes_invariance_gate(0.05, max_flip_rate=0.10).passed is True
    assert passes_invariance_gate(0.05, max_flip_rate=0.01).passed is False


def test_gate_new_elements_reports_one_result_per_key():
    results = gate_new_elements({"topic_domain": 0.01, "surgical_training": 0.05})
    assert results["topic_domain"].passed is True
    assert results["surgical_training"].passed is False


def test_all_pass_requires_every_element_to_clear_the_gate():
    passing = gate_new_elements({"a": 0.0, "b": 0.01})
    failing = gate_new_elements({"a": 0.0, "b": 0.05})
    assert all_pass(passing) is True
    assert all_pass(failing) is False


# ---- L5: gating every feature, not just newly proposed ones ---------------------------------

def test_flip_rates_by_element_covers_every_named_question():
    before = {"a": {"holistic": {"choice": "surgeon"}, "topic": {"noul": 0.9}},
             "b": {"holistic": {"choice": "physician"}, "topic": {"noul": 0.1}}}
    after = {"a": {"holistic": {"choice": "physician"}, "topic": {"noul": 0.9}},
            "b": {"holistic": {"choice": "physician"}, "topic": {"noul": 0.1}}}

    rates = flip_rates_by_element(before, after, ["holistic", "topic"])

    assert rates == {"holistic": pytest.approx(0.5), "topic": 0.0}


def test_flip_rates_by_element_skips_items_missing_that_question():
    before = {"a": {"x": {"noul": 0.9}}, "b": {}}
    after = {"a": {"x": {"noul": 0.1}}, "b": {}}

    rates = flip_rates_by_element(before, after, ["x"])

    assert rates == {"x": 1.0}   # only item "a" has the question at all


def test_elements_over_gate_drops_only_what_exceeds_the_threshold():
    rates = {"holistic": 0.0795, "surgical_training": 0.01, "topic_domain": 0.02}
    assert elements_over_gate(rates) == ["holistic"]


def test_elements_over_gate_is_empty_when_everything_passes():
    assert elements_over_gate({"a": 0.0, "b": 0.02}) == []


def test_elements_over_gate_orders_dropped_names_alphabetically():
    rates = {"z_bad": 0.5, "a_bad": 0.3, "fine": 0.0}
    assert elements_over_gate(rates) == ["a_bad", "z_bad"]
