"""Feature: choosing which item to ask the human about next.

Selection has to be informative (labels are scarce) and stochastic (so the labels
can be inverse-probability-weighted afterwards).
"""
import random
from collections import Counter

import pytest

from jev_flywheel.answers import AnswerCache
from jev_flywheel.items import Item
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.selection import (
    SelectionPolicy,
    answer_entropy,
    binary_entropy,
    build_candidates,
    choose,
    evidence_conflict,
)

CARD = """
name: Card
scores:
  - name: Outcome
    key: outcome
    question_type: noul
    instructions: "Good?"
    elements:
      - {key: a, question_type: noul, instructions: "A?"}
      - {key: b, question_type: noul, instructions: "B?"}
    decision:
      model: multinomial_logistic
      classes: ["yes", "no"]
      features: [self.holistic.logit_p, a.logit_p, b.logit_p]
      parameters:
        weights:
          "yes": {intercept: 0.0, self.holistic.logit_p: 1.0, a.logit_p: 1.0, b.logit_p: 1.0}
"""


def setup(rows):
    """``rows`` maps item id to (holistic p, a p, b p)."""
    card = Scorecard.from_yaml(CARD)
    questions = card.questions()
    cache = AnswerCache()
    for item_id, (holistic, a, b) in rows.items():
        for name, p in (("Outcome", holistic), ("outcome.a", a), ("outcome.b", b)):
            cache.put(item_id, name, questions[name], {"type": "noul", "noul": p})
    items = [Item(id=i, text=i) for i in rows]
    return card.score("Outcome"), questions, cache, items


def candidates_for(rows, labeled=(), policy=SelectionPolicy()):
    score, questions, cache, items = setup(rows)
    return {c.item_id: c for c in build_candidates(
        score, questions, cache, items, set(labeled), policy=policy)}


def test_binary_entropy_is_one_at_a_coin_flip_and_zero_when_certain():
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.999999) < 0.001
    assert binary_entropy(0.9) == pytest.approx(binary_entropy(0.1))


def test_a_confident_agreeing_item_is_a_poor_question():
    got = candidates_for({"sure": (0.95, 0.95, 0.95), "torn": (0.55, 0.45, 0.5)})

    assert got["torn"].components["uncertainty"] > got["sure"].components["uncertainty"]
    assert got["torn"].score > got["sure"].score


def test_disagreement_between_the_head_and_jevs_holistic_answer_is_flagged():
    # Holistic says No (0.3) but the two elements together outvote it.
    got = candidates_for({"override": (0.3, 0.95, 0.95), "agree": (0.9, 0.9, 0.9)})

    assert got["override"].components["disagreement"] == 1.0
    assert got["agree"].components["disagreement"] == 0.0


def test_evidence_that_pulls_both_ways_is_conflict_and_unanimous_evidence_is_not():
    assert evidence_conflict({"a": 2.0, "b": 2.0}) == 0.0
    assert evidence_conflict({"a": 2.0, "b": -2.0}) == pytest.approx(0.5)
    assert evidence_conflict({"a": 3.0, "b": -1.0}) == pytest.approx(0.25)
    assert evidence_conflict({}) == 0.0


def test_an_item_whose_own_inputs_disagree_scores_higher_on_conflict():
    got = candidates_for({"split": (0.9, 0.95, 0.05), "united": (0.9, 0.95, 0.95)})

    assert got["split"].components["conflict"] > got["united"].components["conflict"]


def test_an_item_where_every_answer_is_unsure_is_measured_as_fully_ambiguous():
    got = candidates_for({"murky": (0.5, 0.5, 0.5), "clear": (0.9, 0.9, 0.9)})

    assert got["murky"].components["ambiguity"] > got["clear"].components["ambiguity"]
    assert got["murky"].components["ambiguity"] == pytest.approx(1.0)


def test_ambiguity_is_recorded_but_does_not_change_the_score_by_default():
    # It was a penalty, on the theory that an all-unsure item is irreducibly ambiguous. That
    # was false on the shipped corpus, whose "neutral" tier looks unlearnable under the
    # starting questions but is ~90% recoverable from an undeclared factor. It is kept as a
    # diagnostic so a future policy can use it with evidence.
    assert SelectionPolicy().ambiguity == 0.0

    got = candidates_for({"murky": (0.5, 0.5, 0.5), "clear": (0.9, 0.9, 0.9)})

    assert got["murky"].components["ambiguity"] > got["clear"].components["ambiguity"]
    assert got["murky"].score > got["clear"].score        # uncertainty still drives it


def test_selection_keeps_most_of_the_effective_sample_it_is_given():
    # Sharper picks mean more unequal propensities, and the fit is weighted by their inverse,
    # so a policy can select so keenly that it starves the fit below the ladder's floor.
    # This pins the trade-off: it may be retuned, but not silently.
    import random as _random

    from jev_flywheel.sampling import inverse_propensity_weights, kish_n_effective

    got = candidates_for({f"i{k}": (0.5 + k / 120, 0.5 + k / 200, 0.5) for k in range(60)})
    rng = _random.Random(0)
    propensities = [choose(list(got.values()), rng).propensity for _ in range(40)]

    retained = kish_n_effective(inverse_propensity_weights(propensities)) / 40

    assert retained > 0.6, f"selection kept only {retained:.0%} of its effective sample"


def test_answer_entropy_covers_every_question_type():
    answers = {
        "y": {"type": "noul", "noul": 0.5},
        "c": {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}},
        "s": {"type": "score", "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}},
    }
    assert answer_entropy(answers) == pytest.approx(1.0)
    assert answer_entropy({}) == 0.0


def test_labeled_items_are_never_offered_again():
    got = candidates_for({"a": (0.5, 0.5, 0.5), "b": (0.6, 0.6, 0.6)}, labeled={"a"})

    assert set(got) == {"b"}


def test_an_item_near_one_already_labeled_is_less_novel_than_a_distant_one():
    rows = {"labeled": (0.9, 0.9, 0.9), "twin": (0.9, 0.9, 0.9), "far": (0.1, 0.1, 0.1),
            "mid": (0.5, 0.5, 0.5)}

    got = candidates_for(rows, labeled={"labeled"})

    assert got["twin"].components["novelty"] < got["far"].components["novelty"]


def test_with_nothing_labeled_yet_novelty_is_neutral_rather_than_infinite():
    got = candidates_for({"a": (0.9, 0.9, 0.9), "b": (0.1, 0.1, 0.1)})

    assert {c.components["novelty"] for c in got.values()} == {0.5}


def test_selection_records_the_probability_it_chose_with_so_labels_can_be_weighted():
    got = candidates_for({f"i{k}": (0.5 + k / 40, 0.5, 0.5) for k in range(20)})

    selection = choose(list(got.values()), random.Random(0))

    assert 0 < selection.propensity < 1
    assert selection.pool_size == 20
    assert selection.record()["propensity"] == selection.propensity
    assert selection.record()["selection_policy"] == "selection-v1"


def test_every_item_keeps_a_positive_chance_so_every_label_can_be_inverse_weighted():
    got = candidates_for({f"i{k}": (0.5 + k / 100, 0.5, 0.5) for k in range(30)})
    policy = SelectionPolicy(explore=0.2)
    floor = 0.2 / 30
    seen = Counter()
    rng = random.Random(3)

    for _ in range(2000):
        picked = choose(list(got.values()), rng, policy)
        assert picked.propensity >= floor - 1e-12
        seen[picked.candidate.item_id] += 1

    assert len(seen) > 20          # exploration reaches well beyond the top scorers


def test_a_higher_scoring_item_is_chosen_more_often():
    got = candidates_for({"torn": (0.5, 0.5, 0.5), "conflicted": (0.9, 0.95, 0.05),
                          "sure": (0.99, 0.99, 0.99)})
    rng = random.Random(5)
    picks = Counter(choose(list(got.values()), rng).candidate.item_id for _ in range(3000))

    assert picks["conflicted"] > picks["sure"]


def test_choosing_from_nothing_says_so():
    with pytest.raises(ValueError, match="no unlabeled"):
        choose([], random.Random(0))


def test_the_policy_is_frozen_so_a_recorded_selection_stays_interpretable():
    with pytest.raises(Exception):
        SelectionPolicy().temperature = 9
