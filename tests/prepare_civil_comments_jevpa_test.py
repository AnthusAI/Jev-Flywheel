"""Offline setup specs for the Civil Comments JEVPA workspace."""
from __future__ import annotations

import json

from scripts.prepare_civil_comments_jevpa import prepare
from jev_flywheel.workspace import Workspace


def test_prepare_exposes_only_discovery_labels_to_the_workspace(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    discovery = tmp_path / "discovery.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in [
        {"id": "d", "text": "discovery", "metadata": {"split": "discovery"}},
        {"id": "s", "text": "selection", "metadata": {"split": "selection"}},
        {"id": "t", "text": "test", "metadata": {"split": "test"}},
    ]) + "\n")
    discovery.write_text(json.dumps({"id": "d", "text": "discovery", "metadata": {
        "split": "discovery", "reference_label": 1}}) + "\n")

    prepare(corpus, discovery, tmp_path / "workspace")

    workspace = Workspace(tmp_path / "workspace")
    assert workspace.n_labeled("Toxicity") == 1
    assert workspace.item("s").reference_label is None
    assert workspace.item("t").reference_label is None


def test_prepare_accepts_a_registered_larger_discovery_set(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    discovery = tmp_path / "discovery.jsonl"
    rows = [{"id": f"d{index}", "text": "discovery", "metadata": {"split": "discovery"}}
            for index in range(4)]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    discovery.write_text("\n".join(json.dumps({**row, "metadata": {
        "split": "discovery", "reference_label": index % 2}}) for index, row in enumerate(rows)) + "\n")

    prepare(corpus, discovery, tmp_path / "workspace", expected_discovery_size=4)

    assert Workspace(tmp_path / "workspace").n_labeled("Toxicity") == 4
