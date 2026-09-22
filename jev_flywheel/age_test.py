"""Feature: an age inserted at a bio's first subject pronoun, and the eligibility rule that
keeps the inserted young age from contradicting the bio's own text.

See ``studies/PREREGISTERED.md``, "does the engine read age?" for the rule these specs pin.
"""
from jev_flywheel.age import eligible, insert_age


def test_sentence_initial_pronoun_gets_at_age_prepended_and_lowercased():
    text = "He is currently researching diabetes. He says he enjoys it."
    result = insert_age(text, 61)
    assert result is not None
    assert result.text == "At 61, he is currently researching diabetes. He says he enjoys it."
    assert result.case == "sentence_initial"


def test_mid_sentence_pronoun_gets_comma_at_age_comma_inserted_before_it():
    result = insert_age("In 2003 she joined the clinic.", 34)
    assert result is not None
    assert result.text == "In 2003, at 34, she joined the clinic."
    assert result.case == "mid_sentence"


def test_second_sentence_pronoun_after_period_is_still_sentence_initial():
    text = "He studied medicine. She continued her training."
    result = insert_age(text, 35)
    # first subject pronoun is "He" (sentence-initial)
    assert result.text == "At 35, he studied medicine. She continued her training."
    assert result.case == "sentence_initial"


def test_pronoun_mid_sentence_after_other_words_is_mid_sentence():
    text = "Board certified in 2010 he moved to Boston for his residency."
    result = insert_age(text, 62)
    assert result is not None
    assert result.case == "mid_sentence"
    assert result.text == "Board certified in 2010, at 62, he moved to Boston for his residency."


def test_a_bio_without_a_subject_pronoun_yields_nothing():
    assert insert_age("Board certified in surgery since 2001.", 34) is None


def test_possessives_and_objects_do_not_qualify_as_the_insertion_point():
    assert insert_age("Her research won an award for her.", 34) is None


def test_eligible_requires_a_subject_pronoun():
    assert eligible("Board certified in surgery since 2001.") is False


def test_eligible_bio_with_no_year_and_no_long_duration():
    assert eligible("He is currently researching diabetes and enjoys teaching.") is True


def test_a_year_before_2000_disqualifies():
    assert eligible("He joined the faculty in 1998.") is False


def test_a_year_2000_or_later_does_not_disqualify():
    assert eligible("He joined the faculty in 2005.") is True


def test_a_duration_of_ten_or_more_years_disqualifies():
    assert eligible("He has practiced medicine for 15 years.") is False


def test_a_qualified_long_duration_disqualifies():
    assert eligible("He has over 30 years of experience.") is False
    assert eligible("He has nearly 12 years of experience.") is False


def test_a_short_duration_does_not_disqualify():
    assert eligible("He has 5 years of experience.") is True


def test_a_duration_with_a_plus_sign_disqualifies():
    assert eligible("He has 15+ years of experience.") is False
