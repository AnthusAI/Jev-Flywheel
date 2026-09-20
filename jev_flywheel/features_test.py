"""Feature: Jev answers become named numeric features.

The contract is deterministic and frozen: same answer plus same criteria always
gives the same numbers, with no fitted preprocessing anywhere.
"""
import math

import pytest

from jev_flywheel.features import (
    HOLISTIC,
    available_terms,
    extract_terms,
    split_feature,
)

EPS = 0.01


def logit(p):
    return math.log(p / (1 - p))


def test_a_feature_name_splits_into_its_element_and_term():
    assert split_feature("praise.logit_p") == ("praise", "logit_p")
    assert split_feature("tone.clr.dismissive") == ("tone", "clr.dismissive")


def test_the_holistic_question_and_shared_elements_have_their_own_prefixes():
    # Feature names never contain the owning score's key, so a decision block
    # is portable between scores.
    assert split_feature("self.holistic.logit_p") == (HOLISTIC, "logit_p")
    assert split_feature("shared.transferred.logit_p") == ("shared.transferred", "logit_p")
    assert split_feature("shared.tone.clr.angry") == ("shared.tone", "clr.angry")


def test_a_yes_no_answer_gives_a_clipped_log_odds():
    terms = extract_terms({"noul": 0.9}, "noul", None, EPS)
    assert terms["logit_p"] == pytest.approx(logit(0.9))
    assert terms["p"] == pytest.approx(0.9)
    assert terms["is_yes"] == 1.0


def test_a_yes_no_answer_below_a_half_is_a_no():
    assert extract_terms({"noul": 0.2}, "noul", None, EPS)["is_yes"] == 0.0


def test_extreme_probabilities_are_clipped_so_one_flip_cannot_dominate():
    # Jev's tails are not calibrated and answers flip about 1% of the time.
    # Unclipped, 0.9999 against 0.0001 is an 18-unit swing in log-odds.
    high = extract_terms({"noul": 0.9999}, "noul", None, EPS)["logit_p"]
    assert high == pytest.approx(logit(1 - EPS))


def test_a_choice_answer_gives_a_centered_log_ratio_per_option():
    answer = {"choice": "positive", "probabilities": {"positive": 0.8, "negative": 0.2}}
    terms = extract_terms(answer, "choice", {"positive": None, "negative": None}, EPS)

    expected = math.log(0.8) - (math.log(0.8) + math.log(0.2)) / 2
    assert terms["clr.positive"] == pytest.approx(expected)
    assert terms["clr.negative"] == pytest.approx(-expected)


def test_the_centered_log_ratio_sums_to_zero():
    # This is why CLR is used: per-option logits are collinear, so regularization
    # splits weight between them arbitrarily and refits are not comparable.
    answer = {"choice": "a", "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}}
    terms = extract_terms(answer, "choice", {"a": None, "b": None, "c": None}, EPS)

    assert sum(terms[f"clr.{o}"] for o in "abc") == pytest.approx(0.0)


def test_a_choice_answer_marks_which_option_was_chosen():
    answer = {"choice": "negative", "probabilities": {"positive": 0.4, "negative": 0.6}}
    terms = extract_terms(answer, "choice", {"positive": None, "negative": None}, EPS)

    assert terms["chosen.negative"] == 1.0
    assert terms["chosen.positive"] == 0.0
    assert terms["top_p"] == pytest.approx(0.6)


def test_options_follow_the_configured_order_not_the_answers():
    # An answer may omit options, so a contract keyed on answer order would not
    # be a contract.
    answer = {"choice": "b", "probabilities": {"b": 0.7}}
    terms = extract_terms(answer, "choice", {"a": None, "b": None}, EPS)

    assert terms["chosen.a"] == 0.0
    assert terms["top_p"] == pytest.approx(0.7)


def test_entropy_is_normalized_so_it_compares_across_option_counts():
    flat_two = extract_terms(
        {"choice": "a", "probabilities": {"a": 0.5, "b": 0.5}},
        "choice", {"a": None, "b": None}, EPS)
    flat_three = extract_terms(
        {"choice": "a", "probabilities": {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}},
        "choice", {"a": None, "b": None, "c": None}, EPS)

    assert flat_two["entropy"] == pytest.approx(1.0)
    assert flat_three["entropy"] == pytest.approx(1.0)


def test_a_score_answer_gives_a_normalized_score_and_an_expected_level():
    answer = {"score": 1.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0},
              "confidence": 0.7}
    terms = extract_terms(answer, "score", ["none", "mild", "moderate", "strong"], EPS)

    assert terms["score_norm"] == pytest.approx(1.5 / 3)
    assert terms["expected_level"] == pytest.approx(1.5 / 3)
    assert terms["confidence"] == pytest.approx(0.7)


def test_score_probabilities_work_with_integer_keys():
    # The SDK models score probabilities as dict[int, float]. Code that only
    # looks up string keys works on JSONL-cached answers, where the round trip
    # stringified them, and silently returns all zeros on the live path.
    integer_keyed = {"score": 3.0, "probabilities": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}}
    string_keyed = {"score": 3.0, "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}}
    criteria = ["none", "mild", "moderate", "strong"]

    from_integers = extract_terms(integer_keyed, "score", criteria, EPS)
    from_strings = extract_terms(string_keyed, "score", criteria, EPS)

    assert from_integers == from_strings
    assert from_integers["expected_level"] == pytest.approx(1.0)


def test_a_legend_that_disagrees_with_the_configured_criteria_is_an_error():
    # The rubric changed under us. Guessing an alignment would silently shift
    # every level by one.
    answer = {"score": 1.0, "probabilities": {0: 0.5, 1: 0.5}, "legend": {0: "a", 1: "b"}}

    with pytest.raises(ValueError, match="legend"):
        extract_terms(answer, "score", ["none", "mild", "moderate"], EPS)


def test_available_terms_enumerates_what_each_question_type_can_produce():
    assert available_terms("noul", None) == {"logit_p", "p", "is_yes"}
    assert "clr.positive" in available_terms("choice", {"positive": None, "negative": None})
    assert "chosen.negative" in available_terms("choice", {"positive": None, "negative": None})
    assert "clr.2" in available_terms("score", ["a", "b", "c"])
    assert "expected_level" in available_terms("score", ["a", "b", "c"])


def test_every_extracted_term_was_declared_available():
    # The two functions have to agree or load-time validation is worthless.
    criteria = {"positive": None, "negative": None}
    answer = {"choice": "positive", "probabilities": {"positive": 0.8, "negative": 0.2},
              "confidence": 0.9}

    produced = set(extract_terms(answer, "choice", criteria, EPS))

    assert produced <= available_terms("choice", criteria)


def test_an_unsupported_question_type_is_rejected():
    with pytest.raises(ValueError, match="question type"):
        available_terms("freeform", None)
    with pytest.raises(ValueError, match="question type"):
        extract_terms({}, "freeform", None, EPS)
