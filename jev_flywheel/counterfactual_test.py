"""Feature: gender-swapped counterfactuals.

The swap rule is fixed before the bias study runs, so these specs pin its behaviour: what it
changes, what it leaves alone, and how it resolves the ambiguous ``her``.
"""
import pytest

from jev_flywheel.counterfactual import redact_names, redact_names_batch, swap_gender

try:
    import spacy
    spacy.load("en_core_web_sm")
    _HAS_NAME_MODEL = True
except (ImportError, OSError):
    _HAS_NAME_MODEL = False

requires_name_model = pytest.mark.skipif(
    not _HAS_NAME_MODEL, reason="spacy en_core_web_sm is not installed (pip install '.[bios]')")


def test_pronouns_and_reflexives_flip_both_ways():
    assert swap_gender("He said he did it himself.").text == "She said she did it herself."
    assert swap_gender("She said she did it herself.").text == "He said he did it himself."


def test_case_is_preserved():
    assert swap_gender("HE and He and he").text == "SHE and She and she"


def test_possessive_her_becomes_his():
    swap = swap_gender("Her research on her patients won her an award.")
    assert swap.text == "His research on his patients won him an award."
    assert swap.her_resolved == 3


def test_object_her_before_punctuation_or_preposition_becomes_him():
    assert swap_gender("They asked her.").text == "They asked him."
    assert swap_gender("They asked her to speak.").text == "They asked him to speak."
    assert swap_gender("Working with her, the team grew.").text == "Working with him, the team grew."


def test_his_and_him_collapse_onto_her():
    assert swap_gender("His work made him famous.").text == "Her work made her famous."


def test_role_nouns_and_titles_flip():
    assert swap_gender("Mr. Smith, a father of two, is a spokesman.").text == \
        "Ms. Smith, a mother of two, is a spokeswoman."


def test_names_and_everything_else_are_untouched():
    text = "Emanuel launched USM in 2009; the site is a one-stop shop for agencies."
    swap = swap_gender(text)
    assert swap.text == text
    assert swap.swapped == 0


def test_substrings_of_words_are_not_swapped():
    # "the", "hero", "shell", "themes", "history" all contain pronoun spellings.
    text = "The hero in the shell wrote themes about history."
    assert swap_gender(text).text == text


def test_medical_content_phrases_are_not_swapped():
    swap = swap_gender("She directs the Women's Health clinic and her men's health research.")
    assert swap.text == "He directs the Women's Health clinic and his men's health research."


def test_miss_sir_and_madam_are_swapped():
    assert swap_gender("Miss Jones and Sir John").text == "Mr Jones and Madam John"


def test_swap_count_is_reported():
    assert swap_gender("He and his wife").swapped == 3


@requires_name_model
def test_first_name_in_a_person_span_is_redacted():
    redaction = redact_names("Jennifer has extensive research experience.")
    assert redaction.text == "[name] has extensive research experience."
    assert redaction.redacted == 1


@requires_name_model
def test_possessive_name_keeps_the_apostrophe_s():
    redaction = redact_names("Jennifer's research won an award.")
    assert redaction.text == "[name]'s research won an award."
    assert redaction.redacted == 1


@requires_name_model
def test_surname_alone_is_not_on_the_name_list_so_is_untouched():
    # "Smith" is a surname, not a first name, and is never on fixtures/bios/first_names.txt.
    redaction = redact_names("Dr. Smith is a surgeon.")
    assert redaction.text == "Dr. Smith is a surgeon."
    assert redaction.redacted == 0


@requires_name_model
def test_a_first_name_outside_any_person_span_is_left_alone():
    # "May" is on the first-name list but reads as the month here, not a PERSON entity.
    redaction = redact_names("She graduated in May with honors.")
    assert redaction.redacted == 0


@requires_name_model
def test_batch_matches_the_single_text_path():
    texts = ["Jennifer has extensive research experience.", "Dr. Smith is a surgeon."]
    batched = redact_names_batch(texts)
    singles = [redact_names(t) for t in texts]
    assert [r.text for r in batched] == [r.text for r in singles]
    assert [r.redacted for r in batched] == [r.redacted for r in singles]
