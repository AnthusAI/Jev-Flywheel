"""Feature: gender-swapped counterfactuals.

The swap rule is fixed before the bias study runs, so these specs pin its behaviour: what it
changes, what it leaves alone, and how it resolves the ambiguous ``her``.
"""
from jev_flywheel.counterfactual import swap_gender


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


def test_swap_count_is_reported():
    assert swap_gender("He and his wife").swapped == 3
