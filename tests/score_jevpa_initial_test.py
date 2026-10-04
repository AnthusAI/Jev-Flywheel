"""Initial JEVPA selection fits only discovery feedback and keeps test IDs sealed."""
from scripts.score_jevpa_initial import score_initial


def test_initial_selection_reuses_the_live_cache_without_a_test_label(tmp_path):
    # The real small fixture is the contract here: it carries all answer shapes and propensity.
    report = score_initial(
        "var/jevpa_live/initial/initial-candidates.json", "fixtures/items.jsonl",
        "fixtures/recordings/simulated-labeler/feedback.jsonl", "var/jevpa_live/splits.json",
        "var/jevpa_live/answer_workspace/answers.jsonl", tmp_path)

    assert report["discovery_labels"] == 140
    assert report["selection_items"] == 300
    assert len(report["candidates"]) == 4
    assert all(record["fit_n"] == 140 for record in report["candidates"])
