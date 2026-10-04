"""The offline JEVPA pilot preserves its splits, budget, and answer coverage."""
import gzip
import json
from pathlib import Path

from scripts.run_jevpa_offline import (
    _single_scorecard,
    feasibility_report,
    partition_pool_ids,
    run_pilot,
)
from jev_flywheel.proposal import default_features
from jev_flywheel.scorecard import Scorecard


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_one_recorded_proposal_is_reported_as_a_machinery_pilot(tmp_path):
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "script.json").write_text(json.dumps({
        "score": "Sentiment",
        "steps": [{"op": "steer", "analyst_reply": json.dumps({
            "add_elements": [{"key": "topic_domain"}],
        })}],
    }))
    with gzip.open(recording / "extra_answers.jsonl.gz", "wt", encoding="utf-8") as out:
        out.write(json.dumps({"item_id": "a", "name": "sentiment.topic_domain"}) + "\n")
        out.write(json.dumps({"item_id": "b", "name": "sentiment.topic_domain"}) + "\n")
    write_jsonl(tmp_path / "items.jsonl", [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    write_jsonl(tmp_path / "recording" / "feedback.jsonl", [
        {"item_id": "a", "final_answer_value": "positive"},
        {"item_id": "b", "final_answer_value": "negative"},
    ])

    report = feasibility_report(tmp_path, recording)

    assert report["recorded_proposal_count"] == 1
    assert report["distinct_candidate_sets"] == 1
    assert report["proposed_question_answer_coverage"] == {
        "sentiment.topic_domain": {"answered_items": 2, "dataset_items": 3}
    }
    assert report["result_kind"] == "machinery_pilot_only"
    assert report["three_arm_comparison_possible"] is False


def test_no_proposal_does_not_get_misreported_as_a_negative_result(tmp_path):
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "script.json").write_text(json.dumps({"steps": []}))
    with gzip.open(recording / "extra_answers.jsonl.gz", "wt", encoding="utf-8"):
        pass
    write_jsonl(tmp_path / "items.jsonl", [])
    write_jsonl(recording / "feedback.jsonl", [])

    report = feasibility_report(tmp_path, recording)

    assert report["recorded_proposal_count"] == 0
    assert report["result_kind"] == "feasibility_only"
    assert report["three_arm_comparison_possible"] is False


def test_pool_partition_is_seeded_disjoint_and_reserves_test_after_selection():
    rows = [{"id": f"item-{i:03}", "metadata": {"split": "pool"}}
            for i in range(1300)]

    discovery = {"item-000", "item-001"}
    selection, test = partition_pool_ids(rows, discovery, seed=91,
                                         selection_size=500, test_size=600)

    assert len(selection) == 500
    assert len(test) == 600
    assert set(selection).isdisjoint(test)
    assert set(selection).isdisjoint(discovery)
    assert set(test).isdisjoint(discovery)
    assert (selection, test) == partition_pool_ids(
        rows, discovery, seed=91, selection_size=500, test_size=600)


def test_single_question_candidates_use_the_proposal_default_feature_contract():
    reference = Scorecard.from_yaml(
        Path("fixtures/scorecards/reference_full.yaml").read_text())
    for element in reference.score("Sentiment").elements:
        candidate = _single_scorecard(reference, element.key)
        expected = ["self.holistic.clr.positive"] + default_features(
            element.key, element.question_type, element.criteria)

        assert candidate.score("Sentiment").decision.features == expected


def test_offline_search_uses_complete_shared_splits_and_reports_actual_fit_budget():
    report = run_pilot()

    assert report["protocol"]["discovery_items"] == 140
    assert report["protocol"]["selection_items"] == 500
    assert report["protocol"]["exploratory_test_items"] == 600
    assert report["network_calls"] == report["analyst_calls"] == 0
    assert 7 <= report["greedy"]["candidate_fits"] <= 9
    assert 7 <= report["jevpa"]["candidate_fits"] <= 9
    assert len(report["candidate_fits"]) >= report["greedy"]["candidate_fits"]
    assert all(
        coverage["selection_answered"] == coverage["selection_items"]
        and coverage["test_answered"] == coverage["test_items"]
        for coverage in report["answer_coverage"].values()
    )
    assert report["greedy"]["exploratory_test"]["n"] == 600
    assert report["jevpa"]["exploratory_test"]["n"] == 600
    assert isinstance(report["merge_differs_from_greedy"], bool)
    if report["jevpa"]["candidate_fits"] < 9:
        assert report["jevpa_budget_unused_reason"]
    assert report["evidence_status"] == "exploratory_scripted_reference_labels"
