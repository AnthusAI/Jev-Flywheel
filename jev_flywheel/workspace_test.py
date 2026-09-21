"""Feature: the workspace keeps the flywheel's history as append-only files."""
import gzip
import json
import shutil
from pathlib import Path

import pytest

from jev_flywheel.items import FeedbackItem
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.workspace import Workspace, WorkspaceError

REAL_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

REFERENCE = """
name: Sentiment
scores:
  - name: Sentiment
    key: sentiment
    question_type: choice
    instructions: "Sentiment?"
    criteria: {positive: null, negative: null}
    elements:
      - {key: praise, question_type: noul, instructions: "Praise?"}
"""

V1 = """
name: Sentiment
scores:
  - name: Sentiment
    key: sentiment
    question_type: choice
    instructions: "Sentiment?"
    criteria: {positive: null, negative: null}
    decision:
      model: linear_threshold
      classes: ["positive", "negative"]
      positive_class: "positive"
      features: [self.holistic.clr.positive]
      parameters: {weights: {intercept: 0.0, self.holistic.clr.positive: 2.0}}
"""


@pytest.fixture
def fixtures(tmp_path):
    """A miniature stand-in for fixtures/: three items and their cached answers."""
    root = tmp_path / "fixtures"
    (root / "scorecards").mkdir(parents=True)
    (root / "scorecards" / "reference_full.yaml").write_text(REFERENCE)
    (root / "scorecards" / "v1.yaml").write_text(V1)
    items = [{"id": f"i{n}", "text": f"text {n}",
              "metadata": {"split": "pool" if n < 2 else "test", "reference_label": "positive"}}
             for n in range(3)]
    (root / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    answers = [{"id": f"i{n}", "model": "jev-test", "answers": {
        "Sentiment": {"type": "choice", "choice": "positive", "confidence": 0.9,
                      "probabilities": {"positive": 0.9, "negative": 0.1}},
        "sentiment.praise": {"type": "noul", "noul": 0.8}}} for n in range(3)]
    with gzip.open(root / "answers.jsonl.gz", "wt") as handle:
        handle.write("\n".join(json.dumps(a) for a in answers) + "\n")
    return root


@pytest.fixture
def workspace(tmp_path, fixtures):
    return Workspace.init(tmp_path / "var", fixtures)


def record(item_id, final, initial="positive", comment=None, score="Sentiment", **meta):
    return FeedbackItem(id=f"f-{item_id}-{final}", item_id=item_id, score_name=score,
                        initial_answer_value=initial, final_answer_value=final,
                        edit_comment_value=comment, metadata={"propensity": 0.5, **meta})


def test_a_fresh_workspace_is_seeded_from_the_fixtures_with_no_network(workspace):
    assert workspace.exists
    assert [i.id for i in workspace.items] == ["i0", "i1", "i2"]
    assert workspace.version == 1
    assert workspace.scorecard().score("Sentiment").decision.model == "linear_threshold"


def test_the_cached_answers_are_imported_so_the_seed_scorecard_serves_immediately(workspace):
    result = workspace.predict("i0", "Sentiment")

    assert result.value == "positive"
    assert result.confidence == pytest.approx(0.9, abs=1e-6)


def test_a_workspace_says_which_engine_answers_it_and_defaults_to_jev(workspace):
    assert workspace.engine == "jev"
    workspace.manifest_path.unlink()          # a workspace made before engines existed
    assert workspace.engine == "jev"


def test_a_workspace_can_be_seeded_from_another_engines_answers(tmp_path, fixtures):
    laya = [{"id": f"i{n}", "model": "laya:test", "answers": {
        "Sentiment": {"type": "choice", "choice": "negative", "confidence": 0.6,
                      "probabilities": {"positive": 0.4, "negative": 0.6}},
        "sentiment.praise": {"type": "noul", "noul": 0.2}}} for n in range(3)]
    with gzip.open(fixtures / "answers-laya.jsonl.gz", "wt") as handle:
        handle.write("\n".join(json.dumps(a) for a in laya) + "\n")

    other = Workspace.init(tmp_path / "laya", fixtures, answers="answers-laya.jsonl.gz",
                           engine="laya")

    assert other.engine == "laya"
    assert other.cache.model == "laya:test"
    assert other.predict("i0", "Sentiment").value == "negative"     # not Jev's "positive"


def test_the_corpus_splits_are_available_by_name(workspace):
    assert [i.id for i in workspace.split("pool")] == ["i0", "i1"]
    assert [i.id for i in workspace.split("test")] == ["i2"]


def test_a_missing_workspace_says_what_to_do(tmp_path):
    with pytest.raises(WorkspaceError, match="flywheel init"):
        Workspace(tmp_path / "nowhere").require()


def test_initializing_over_an_existing_workspace_is_refused_without_force(workspace, fixtures):
    with pytest.raises(WorkspaceError, match="--force"):
        Workspace.init(workspace.root, fixtures)


def test_forcing_replaces_the_workspace_and_discards_its_labels(workspace, fixtures):
    workspace.add_feedback(record("i0", "positive"))

    fresh = Workspace.init(workspace.root, fixtures, force=True)

    assert fresh.feedback() == []


def test_feedback_is_appended_in_the_order_it_was_given(workspace):
    workspace.add_feedback(record("i0", "positive"))
    workspace.add_feedback(record("i1", "negative"))

    assert [f.item_id for f in workspace.feedback()] == ["i0", "i1"]


def test_labeled_items_are_reported_so_selection_does_not_repeat_them(workspace):
    workspace.add_feedback(record("i0", "positive"))

    assert workspace.labeled_ids("Sentiment") == {"i0"}
    assert workspace.labeled_ids("Other") == set()


def test_a_revisited_item_counts_once(workspace):
    workspace.add_feedback(record("i0", "positive"))
    workspace.add_feedback(record("i0", "negative"))

    assert workspace.n_labeled("Sentiment") == 1


def test_a_skipped_item_is_labeled_for_selection_but_carries_no_usable_label(workspace):
    workspace.add_feedback(FeedbackItem(
        id="skip", item_id="i0", score_name="Sentiment", label_source="unresolved",
        metadata={"skipped": True}))

    assert "i0" in workspace.labeled_ids("Sentiment")
    assert workspace.n_labeled("Sentiment") == 0


def test_a_new_scorecard_version_records_its_parent_and_never_overwrites(workspace):
    card = workspace.scorecard()
    before = (workspace.scorecards_dir / "v1.yaml").read_text()

    version = workspace.commit_scorecard(card, kind="fit", provenance={"fit_id": "dh-1"})

    assert version == 2
    assert (workspace.scorecards_dir / "v1.yaml").read_text() == before
    entry = workspace.lineage()[-1]
    assert (entry["version"], entry["parent"], entry["kind"]) == (2, 1, "fit")
    assert entry["provenance"] == {"fit_id": "dh-1"}


def test_any_earlier_version_can_be_read_back(workspace):
    workspace.commit_scorecard(workspace.scorecard(), kind="fit")

    assert workspace.scorecard(1).version == 1
    assert workspace.scorecard(2).version == 2
    with pytest.raises(WorkspaceError, match="version 9"):
        workspace.scorecard(9)


def test_committing_an_invalid_scorecard_is_refused_before_anything_is_written(workspace):
    card = workspace.scorecard()
    card.scores[0].decision.features = ["nonexistent.logit_p"]

    with pytest.raises(ValueError):
        workspace.commit_scorecard(card, kind="fit")

    assert workspace.version == 1
    assert not (workspace.scorecards_dir / "v2.yaml").exists()


def test_events_are_recorded_with_the_counts_steering_needs_later(workspace):
    workspace.add_feedback(record("i0", "positive"))
    workspace.add_feedback(record("i1", "negative"))

    event = workspace.log_event("fit", "Sentiment", fitted=True, log_loss=0.5)

    assert event["n_labeled"] == 2 and event["n_feedback"] == 2 and event["version"] == 1
    assert workspace.events("fit") == [event]
    assert workspace.events("rethink") == []


def test_a_fresh_workspace_has_no_steering_history(workspace):
    state = workspace.steering_state("Sentiment")

    assert (state.n_labeled, state.labels_since_fit, state.oof_accuracy) == (0, 0, None)
    assert state.fit_log_losses == []


def test_labels_since_the_last_fit_count_from_that_fit(workspace):
    workspace.add_feedback(record("i0", "positive"))
    workspace.log_event("fit", "Sentiment", fitted=True, log_loss=0.6, oof_accuracy=0.8)
    workspace.add_feedback(record("i1", "negative"))

    state = workspace.steering_state("Sentiment")

    assert state.n_labeled == 2
    assert state.labels_since_fit == 1
    assert state.oof_accuracy == 0.8
    assert state.fit_log_losses == [0.6]


def test_only_fits_that_actually_fitted_contribute_a_loss(workspace):
    workspace.log_event("fit", "Sentiment", fitted=False)
    workspace.log_event("fit", "Sentiment", fitted=True, log_loss=0.4, oof_accuracy=0.9)

    assert workspace.steering_state("Sentiment").fit_log_losses == [0.4]


def test_explained_disagreements_are_counted_since_the_last_rethink(workspace):
    workspace.add_feedback(record("i0", "negative", comment="sarcasm"))       # explained mismatch
    workspace.add_feedback(record("i1", "negative"))                          # mismatch, no comment
    workspace.log_event("rethink", "Sentiment", decision="no_change_proposed")
    workspace.add_feedback(record("i2", "negative", comment="irony again"))   # after the rethink

    state = workspace.steering_state("Sentiment")

    assert state.commented_mismatches_since_rethink == 1
    assert state.labels_since_rethink == 1


def test_a_comment_on_an_agreement_is_not_a_mismatch(workspace):
    workspace.add_feedback(record("i0", "positive", comment="fine, good call"))

    assert workspace.steering_state("Sentiment").commented_mismatches_since_rethink == 0


def test_events_for_another_score_do_not_leak_into_this_ones_steering(workspace):
    workspace.log_event("fit", "Other", fitted=True, log_loss=0.1, oof_accuracy=0.99)

    state = workspace.steering_state("Sentiment")

    assert state.fit_log_losses == [] and state.oof_accuracy is None


def test_the_real_fixtures_initialize_a_workspace_offline(tmp_path):
    real = Workspace.init(tmp_path / "var", REAL_FIXTURES)

    assert len(real.items) == 8801
    assert real.predict(real.split("test")[0].id, "Sentiment").value in {"positive", "negative"}
