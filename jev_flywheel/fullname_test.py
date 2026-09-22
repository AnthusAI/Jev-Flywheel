"""Feature: full-name counterfactuals for the second race attempt.

Pins where each of the three insertion kinds fires (pronoun, placeholder, PERSON-span surname)
and that a bio with none of them is excluded. See ``studies/PREREGISTERED.md``, "race from a
full name, second attempt".
"""
from jev_flywheel.fullname import (
    analyze_full_name, apply_full_name, apply_full_name_full, render_full_name)


def test_the_first_subject_pronoun_becomes_the_full_name():
    text = "He is currently researching diabetes. He says he enjoys it."
    assert (apply_full_name(text, "Jamal", "Watson")
            == "Jamal Watson is currently researching diabetes. He says he enjoys it.")


def test_every_placeholder_becomes_the_first_name():
    text = "He won the [name] Borst Award and the [name] Prize."
    result = apply_full_name_full(text, "Greg", "Nowak")
    assert result.placeholders == 2
    assert "[name]" not in result.text
    assert "Greg Borst Award" in result.text
    assert "Greg Prize" in result.text


def test_a_surname_token_inside_a_person_span_becomes_the_surname():
    text = "Dr. Moyer leads the clinic. He trained under Dr. Moyer."
    result = apply_full_name_full(text, "Emily", "Chen")
    assert result.surnames >= 1
    assert "Moyer" not in result.text


def test_a_bio_with_no_insertion_point_returns_none():
    text = "Board certified in surgery since 2001."
    assert apply_full_name(text, "Greg", "Nowak") is None


def test_reports_which_insertion_kinds_fired():
    text = "He is a surgeon."
    result = apply_full_name_full(text, "Greg", "Nowak")
    assert result.pronoun is True
    assert result.placeholders == 0


def test_pronoun_insertion_uses_first_and_last_together():
    result = apply_full_name_full("She runs a practice.", "Emily", "Chen")
    assert result.text.startswith("Emily Chen runs")


def test_analyze_then_render_matches_the_single_call_api():
    text = "He won the [name] Borst Award. Dr. Moyer trained him."
    plan = analyze_full_name(text)
    a = render_full_name(plan, "Greg", "Nowak")
    b = apply_full_name_full(text, "Greg", "Nowak")
    assert a == b


def test_render_full_name_is_cheap_across_many_names_from_one_plan():
    plan = analyze_full_name("He is a surgeon. [name] trained under Dr. Moyer.")
    first = render_full_name(plan, "Greg", "Nowak")
    second = render_full_name(plan, "Jamal", "Watson")
    assert first.text != second.text
    assert first.pronoun and second.pronoun
    assert first.placeholders == second.placeholders == plan.n_placeholders


def test_analyze_full_name_reports_no_insertion_point_for_a_bare_bio():
    plan = analyze_full_name("Board certified in surgery since 2001.")
    assert not plan.has_insertion_point
    assert render_full_name(plan, "Greg", "Nowak").text is None
