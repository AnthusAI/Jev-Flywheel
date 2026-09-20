"""Feature: the committed fixtures reproduce the published baseline offline.

These run against the real 8,801-item corpus and its cached Jev answers. They are
the guard that a clone of this repo, with no API keys, sees the same numbers the
lab notes report. If one of them moves, either the fixtures or the pipeline
changed, and the README's claims need re-checking.
"""
from pathlib import Path

import pytest

from jev_flywheel.answers import AnswerCache, import_answers_jsonl
from jev_flywheel.head import decide
from jev_flywheel.items import agrees, load_items
from jev_flywheel.scorecard import Scorecard

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def items():
    return load_items(FIXTURES / "items.jsonl")


@pytest.fixture(scope="module")
def reference():
    return Scorecard.from_yaml((FIXTURES / "scorecards" / "reference_full.yaml").read_text())


@pytest.fixture(scope="module")
def v1():
    return Scorecard.from_yaml((FIXTURES / "scorecards" / "v1.yaml").read_text())


@pytest.fixture(scope="module")
def cache(reference):
    cache = AnswerCache()
    import_answers_jsonl(FIXTURES / "answers.jsonl.gz", reference.questions(), cache)
    return cache


def test_the_corpus_has_the_published_split_sizes(items):
    by_split = {}
    for item in items:
        by_split[item.split] = by_split.get(item.split, 0) + 1

    assert by_split == {"pool": 5280, "test": 3521}


def test_every_item_carries_a_reference_label_and_a_tier(items):
    assert all(item.reference_label in {"positive", "negative"} for item in items)
    assert {item.metadata["tier"] for item in items} == {"strong", "medium", "weak", "neutral"}


def test_every_item_has_a_complete_set_of_cached_answers(items, reference, cache):
    questions = reference.questions()

    assert all(cache.answers_for(item.id, questions) is not None for item in items)


def test_the_seven_element_questions_are_all_cache_hits(items, reference, cache):
    # The fixture was collected under exactly these question bodies. If this
    # fails, someone reworded a reference question and orphaned every answer.
    assert cache.plan([item.id for item in items], reference.questions()).is_free


def test_the_v1_scorecard_asks_a_strict_subset_so_it_costs_nothing_to_serve(items, v1, reference):
    assert set(v1.questions()) < set(reference.questions())


def _classify(score, cache, questions, item):
    answers = cache.answers_for(item.id, questions)
    value, confidence, _ = decide(score.feature_vector(answers), score.decision.head())
    return value, confidence, answers


def test_the_v1_head_reproduces_jevs_own_choice(items, v1, cache):
    # A weight of 2.0 on the two-option centered log-ratio equals the log-odds, so
    # the hand-written v1 head must say what Jev says. Exact ties are the one
    # place a choice is arbitrary, so allow those and nothing else.
    score, questions = v1.score("Sentiment"), v1.questions()
    disagreements = 0
    for item in items:
        value, _, answers = _classify(score, cache, questions, item)
        jev = answers["Sentiment"]
        probabilities = jev["probabilities"]
        if abs(probabilities["positive"] - probabilities["negative"]) < 1e-9:
            continue
        disagreements += value != jev["choice"]

    assert disagreements == 0


def test_the_v1_head_reproduces_jevs_own_probability(items, v1, cache):
    score, questions = v1.score("Sentiment"), v1.questions()
    # Jev's tails are clipped at 0.01 by the feature contract, so compare on
    # items whose probabilities are inside the clip.
    checked = 0
    for item in items:
        value, confidence, answers = _classify(score, cache, questions, item)
        top = answers["Sentiment"]["probabilities"][value]
        if 0.02 < top < 0.98:
            assert confidence == pytest.approx(top, abs=1e-6)
            checked += 1

    assert checked > 100


def test_jevs_holistic_accuracy_on_the_test_split_matches_the_published_baseline(items, v1, cache):
    # The lab notes report 0.760 on the 3,521-item test split. This is the number
    # every improvement in this project is measured against.
    score, questions = v1.score("Sentiment"), v1.questions()
    test = [item for item in items if item.split == "test"]

    correct = sum(agrees(_classify(score, cache, questions, item)[0], item.reference_label)
                  for item in test)

    assert correct / len(test) == pytest.approx(0.760, abs=0.005)
