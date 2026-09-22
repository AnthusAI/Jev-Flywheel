"""Feature: race-associated names inserted at a bio's first subject pronoun.

The rule is fixed before the race study runs; these specs pin where the name goes, what is
left alone, and that the control pair is two *different* white names.
"""
import random

import pytest

from jev_flywheel.names import NAMES, insert_name, name_versions


def test_the_first_subject_pronoun_becomes_the_name():
    text = "He is currently researching diabetes. He says he enjoys it."
    assert insert_name(text, "Jamal") == "Jamal is currently researching diabetes. He says he enjoys it."


def test_lowercase_and_female_pronouns_qualify():
    assert insert_name("In 2003 she joined the clinic.", "Emily") == "In 2003 Emily joined the clinic."


def test_possessives_and_objects_do_not_qualify():
    assert insert_name("Her research won an award for her.", "Emily") is None


def test_pronoun_inside_a_word_is_not_matched():
    # "The", "shell", "hero" contain he/she spellings and must not be touched.
    assert insert_name("The shell of the hero.", "Greg") is None


def test_versions_use_two_different_white_names_and_match_gender():
    versions = name_versions("He runs a practice.", "male", random.Random(0))
    assert versions is not None
    assert versions.names["white_a"] != versions.names["white_b"]
    assert versions.names["white_a"] in NAMES["white"]["male"]
    assert versions.names["white_b"] in NAMES["white"]["male"]
    assert versions.names["black"] in NAMES["black"]["male"]
    assert versions.black.startswith(versions.names["black"] + " runs")


def test_versions_are_reproducible_from_the_seed():
    a = name_versions("She runs a practice.", "female", random.Random(7))
    b = name_versions("She runs a practice.", "female", random.Random(7))
    assert a == b


def test_a_bio_without_a_subject_pronoun_yields_nothing():
    assert name_versions("Board certified in surgery since 2001.", "male", random.Random(0)) is None


def test_unknown_gender_is_refused():
    with pytest.raises(ValueError):
        name_versions("He runs a practice.", "other", random.Random(0))
