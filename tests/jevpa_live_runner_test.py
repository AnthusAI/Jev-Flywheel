"""The live JEVPA runner plans without calls and protects held-out splits."""
import json
import hashlib
from pathlib import Path

import pytest

from jev_flywheel.answers import AnswerCache
from jev_flywheel.items import Item
from scripts.run_jevpa_live import (
    frozen_winner_questions, load_items, load_manifest_items, make_plan,
)


def test_plan_is_zero_spend_and_accounts_for_missing_answers(tmp_path):
    cache = AnswerCache(tmp_path / "answers.jsonl")
    questions = {"sentiment.holistic": {"question": "How positive?"},
                 "sentiment.topic": {"question": "What topic?"}}
    cache.put("a", "sentiment.holistic", questions["sentiment.holistic"], {"value": "x"})
    items = [Item("a", "text", metadata={"split": "discovery"}),
             Item("b", "text", metadata={"split": "discovery"})]

    report = make_plan(items, questions, cache, split="discovery", model="jev-model",
                       input_rate=1.0, output_rate=2.0,
                       avg_input_tokens=500, avg_output_tokens=100)

    assert report["mode"] == "plan_only"
    assert report["spend_authorized"] is False
    assert report["planned_requests"] == 2
    assert report["planned_missing_answers"] == 3
    assert report["pricing"]["estimated_cost_usd"] == pytest.approx(0.0014)
    assert report["actual"] is None


def test_loader_refuses_held_out_and_filters_explicit_split(tmp_path):
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(json.dumps(row) for row in [
        {"id": "a", "text": "one", "metadata": {"split": "discovery"}},
        {"id": "b", "text": "two", "metadata": {"split": "test"}},
    ]) + "\n")

    assert [item.id for item in load_items(items_path, "discovery")] == ["a"]
    with pytest.raises(ValueError, match="held-out"):
        load_items(items_path, "test")
    assert [item.id for item in load_items(items_path, "test", final_evaluation=True)] == ["b"]


def test_frozen_winner_manifest_verifies_checksum_and_request_budget(tmp_path):
    card = tmp_path / "winner.yaml"
    card.write_text(Path("fixtures/scorecards/v1.yaml").read_text())
    digest = hashlib.sha256(card.read_bytes()).hexdigest()
    manifest = tmp_path / "winners.json"
    manifest.write_text(json.dumps({
        "requests_used_before_final": 1320,
        "winners": [{"arm": "greedy", "scorecard": "winner.yaml", "sha256": digest}],
    }))

    questions, audit = frozen_winner_questions(manifest)

    assert questions
    assert audit["remaining_request_budget"] == 500
    assert audit["winners"][0]["sha256"] == digest
    card.write_text(card.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        frozen_winner_questions(manifest)


def test_frozen_winner_manifest_can_use_a_preregistered_larger_cap(tmp_path):
    card = tmp_path / "winner.yaml"
    card.write_text(Path("fixtures/scorecards/v1.yaml").read_text())
    digest = hashlib.sha256(card.read_bytes()).hexdigest()
    manifest = tmp_path / "winners.json"
    manifest.write_text(json.dumps({
        "requests_used_before_final": 1460,
        "winners": [{"arm": "greedy", "scorecard": "winner.yaml", "sha256": digest}],
    }))

    _, audit = frozen_winner_questions(manifest, request_cap=1960)

    assert audit["remaining_request_budget"] == 500


def test_manifest_loader_uses_only_frozen_selection_ids(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(json.dumps(row) for row in [
        {"id": "a", "text": "one", "metadata": {"split": "pool"}},
        {"id": "b", "text": "two", "metadata": {"split": "pool"}},
        {"id": "c", "text": "three", "metadata": {"split": "test"}},
    ]) + "\n")
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"discovery_ids": ["a"], "selection_ids": ["b"],
                                    "test_ids": ["c"]}))

    picked = load_manifest_items(items, manifest, ["discovery_ids", "selection_ids"])

    assert [item.id for item in picked] == ["a", "b"]
    with pytest.raises(ValueError, match="held-out"):
        load_manifest_items(items, manifest, ["test_ids"])
