"""Specs for retaining and combining useful JEVPA candidates."""
from jev_flywheel.jevpa import (
    Candidate, adaptive_search_plan, has_distinct_slice_specialists,
    fresh_split_ids, merge_question_sets, pareto_pool,
)
from jev_flywheel.scorecard import Scorecard


def card(elements=()):
    return Scorecard.from_config({
        "name": "demo", "scores": [{
            "name": "Quality", "question_type": "noul", "instructions": "Assess quality",
            "elements": [
                {"key": key, "question_type": "noul", "instructions": f"Check {key}"}
                for key in elements
            ],
            "decision": {"model": "multinomial_logistic", "classes": ["no", "yes"],
                         "features": ["self.holistic.logit_p"]},
        }],
    })


def candidate(name, slices, effective=None, overall=0.2, eligible=True, elements=()):
    return Candidate(
        candidate_id=name, scorecard=card(elements), slice_brier=slices,
        slice_effective_n=effective or {key: 20 for key in slices},
        overall_brier=overall, overall_eligible=eligible,
    )


def test_pareto_pool_keeps_candidates_that_win_different_slices():
    left = candidate("left", {"clear": 0.1, "ambiguous": 0.4})
    right = candidate("right", {"clear": 0.3, "ambiguous": 0.2})
    dominated = candidate("dominated", {"clear": 0.4, "ambiguous": 0.5})

    retained = pareto_pool([dominated, right, left], max_size=4, min_slice_effective_n=10)

    assert {item.candidate_id for item in retained} == {"left", "right"}


def test_adaptive_search_uses_four_shared_two_greedy_and_two_pareto_calls():
    plan = adaptive_search_plan()

    assert len(plan.calls) == 8
    assert [call.phase for call in plan.calls] == [
        "shared_initial", "shared_initial", "shared_initial", "shared_initial",
        "greedy", "greedy", "pareto", "pareto",
    ]
    assert [call.index for call in plan.calls] == list(range(1, 9))
    assert plan.jev_request_cap == 2_000


def test_pareto_mechanism_requires_distinct_unique_slice_specialists():
    positive = candidate("positive", {"positive": 0.05, "negative": 0.7})
    negative = candidate("negative", {"positive": 0.7, "negative": 0.05})
    same_winner = candidate("same", {"positive": 0.05, "negative": 0.8})

    assert has_distinct_slice_specialists([positive, negative])
    assert not has_distinct_slice_specialists([positive, same_winner])
    assert not has_distinct_slice_specialists([positive])


def test_pareto_mechanism_does_not_use_an_underpowered_specialist():
    positive = candidate("positive", {"positive": 0.05, "negative": 0.7},
                         effective={"positive": 60, "negative": 60})
    negative = candidate("negative", {"positive": 0.7, "negative": 0.05},
                         effective={"positive": 60, "negative": 4})

    assert not has_distinct_slice_specialists(
        [positive, negative], min_slice_effective_n=50)


def test_a_fresh_split_excludes_prior_study_ids_and_is_seeded():
    selection, test = fresh_split_ids(
        [f"item-{number}" for number in range(20)],
        excluded_ids={"item-0", "item-1", "item-2"}, seed=20260925,
        selection_size=6, test_size=7)

    assert len(selection) == 6
    assert len(test) == 7
    assert not (set(selection) | set(test)) & {"item-0", "item-1", "item-2"}
    assert set(selection).isdisjoint(test)
    assert (selection, test) == fresh_split_ids(
        [f"item-{number}" for number in range(20)],
        excluded_ids={"item-0", "item-1", "item-2"}, seed=20260925,
        selection_size=6, test_size=7)


def test_pareto_pool_drops_ineligible_and_underpowered_slice_results():
    eligible = candidate("eligible", {"clear": 0.3, "ambiguous": 0.3})
    ineligible = candidate("ineligible", {"clear": 0.01, "ambiguous": 0.01}, eligible=False)
    underpowered = candidate("underpowered", {"clear": 0.01, "ambiguous": 0.01},
                             effective={"clear": 20, "ambiguous": 4})

    retained = pareto_pool([underpowered, ineligible, eligible], max_size=4,
                            min_slice_effective_n=10)

    assert [item.candidate_id for item in retained] == ["eligible"]


def test_pareto_pool_rejects_incomparable_slice_schemas():
    clear = candidate("clear", {"clear": 0.2})
    ambiguous = candidate("ambiguous", {"ambiguous": 0.2})

    try:
        pareto_pool([clear, ambiguous], max_size=2, min_slice_effective_n=10)
    except ValueError as error:
        assert "slice" in str(error)
    else:
        raise AssertionError("a frontier needs the same objectives for every candidate")


def test_pareto_pool_is_deterministic_and_bounded_while_preserving_overall_champion():
    champion = candidate("champion", {"clear": 0.3, "ambiguous": 0.3}, overall=0.1)
    candidates = [champion] + [candidate(f"c{i}", {"clear": i / 10, "ambiguous": 0.5})
                               for i in range(1, 8)]

    first = pareto_pool(candidates, max_size=3, min_slice_effective_n=10)
    second = pareto_pool(list(reversed(candidates)), max_size=3, min_slice_effective_n=10)

    assert len(first) <= 3
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert all(item.overall_eligible for item in first)


def test_a_capped_pool_preserves_distinct_slice_winners_before_balanced_candidates():
    positive = candidate("positive", {"positive": 0.05, "negative": 0.7})
    negative = candidate("negative", {"positive": 0.7, "negative": 0.05})
    balanced = candidate("balanced", {"positive": 0.3, "negative": 0.3})

    retained = pareto_pool([balanced, negative, positive], max_size=2,
                           min_slice_effective_n=10)

    assert {item.candidate_id for item in retained} == {"positive", "negative"}


def test_merge_question_sets_unions_new_questions_and_resets_fitted_numbers():
    incumbent = card(("detail",))
    incumbent_score = incumbent.score("Quality")
    incumbent_score.decision.weights = {"yes": {"self.holistic.logit_p": 2.0}, "no": {}}
    donor = card(("clarity", "evidence"))

    merged = merge_question_sets(incumbent, donor, "Quality")
    score = merged.score("Quality")

    assert {element.key for element in score.elements} == {"detail", "clarity", "evidence"}
    assert "clarity.logit_p" in score.decision.features
    assert "evidence.logit_p" in score.decision.features
    assert score.decision.weights == {}
    assert score.decision.calibration is None
    assert score.decision.provenance is None


def test_merge_rejects_same_key_with_different_question_definition():
    first, second = card(("detail",)), card(("detail",))
    second.score("Quality").elements[0].instructions = "A different check"

    try:
        merge_question_sets(first, second, "Quality")
    except ValueError as error:
        assert "detail" in str(error)
    else:
        raise AssertionError("incompatible questions must not merge silently")
