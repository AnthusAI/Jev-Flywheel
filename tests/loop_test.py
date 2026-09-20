"""Feature: the flywheel loop, end to end on the real corpus, offline.

The human is simulated by the corpus's own reference labels. That is enough to check
the machinery -- selection, labeling, refit, promotion, steering -- without a person.
"""
import gzip
import json
import random
import shutil
from pathlib import Path

import pytest

from jev_flywheel.items import normalize_label
from jev_flywheel.loop import AGREE, DISAGREE, SKIP, next_question, record_label, refit, status
from jev_flywheel.workspace import Workspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SCORE = "Sentiment"


def miniature(target: Path, n_pool: int = 600, n_test: int = 200, seed: int = 0) -> Path:
    """A slice of the real fixtures, so selection over the pool stays fast.

    Every label re-scores the whole unlabeled pool, which is 0.3 s over the full
    5,280 items and a few hundredths of a second over 600. The behavior under test
    does not depend on pool size, so the specs use the small one.
    """
    rng = random.Random(seed)
    rows = [json.loads(line) for line in (FIXTURES / "items.jsonl").read_text().splitlines()]
    pool = [r for r in rows if r["metadata"]["split"] == "pool"]
    test = [r for r in rows if r["metadata"]["split"] == "test"]
    keep = rng.sample(pool, n_pool) + rng.sample(test, n_test)
    ids = {r["id"] for r in keep}
    (target / "scorecards").mkdir(parents=True)
    for name in ("reference_full.yaml", "v1.yaml"):
        shutil.copyfile(FIXTURES / "scorecards" / name, target / "scorecards" / name)
    (target / "items.jsonl").write_text("\n".join(json.dumps(r) for r in keep) + "\n")
    with gzip.open(FIXTURES / "answers.jsonl.gz", "rt") as source, \
            gzip.open(target / "answers.jsonl.gz", "wt") as out:
        for line in source:
            if json.loads(line)["id"] in ids:
                out.write(line)
    return target


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("template")
    return Workspace.init(root / "var", miniature(root / "fixtures"))


@pytest.fixture
def workspace(template, tmp_path):
    shutil.copytree(template.root, tmp_path / "var")
    return Workspace(tmp_path / "var")


def label_like_a_human(workspace, count, seed=0, comment=None):
    """Answer ``count`` questions using the corpus's own labels."""
    rng = random.Random(seed)
    asked = []
    for _ in range(count):
        question = next_question(workspace, SCORE, rng)
        truth = question.item.reference_label
        if normalize_label(truth) == normalize_label(question.result.value):
            record_label(workspace, question, AGREE)
        else:
            record_label(workspace, question, DISAGREE, correct_label=truth, comment=comment)
        asked.append(question)
    return asked


def test_the_next_question_is_an_unlabeled_pool_item_with_a_prediction(workspace):
    question = next_question(workspace, SCORE, random.Random(0))

    assert question.item.split == "pool"
    assert question.result.value in {"positive", "negative"}
    assert 0.0 < question.result.confidence <= 1.0
    assert question.classes == ["positive", "negative"]
    assert "Sentiment" in question.answers


def test_the_test_split_is_never_offered_to_the_human(workspace):
    # It is the scoreboard. If a human labels it, accuracy measured there means nothing.
    offered = label_like_a_human(workspace, 60)

    assert {q.item.split for q in offered} == {"pool"}


def test_an_item_is_never_asked_about_twice(workspace):
    offered = label_like_a_human(workspace, 40)

    ids = [q.item.id for q in offered]
    assert len(ids) == len(set(ids))


def test_agreeing_records_the_prediction_as_the_label_with_its_selection_propensity(workspace):
    question = next_question(workspace, SCORE, random.Random(1))

    record = record_label(workspace, question, AGREE, comment="Yes, this one is clear.")

    assert record.initial_answer_value == record.final_answer_value == question.result.value
    assert record.is_agreement is True
    assert record.edit_comment_value == "Yes, this one is clear."
    assert 0 < record.propensity < 1
    assert record.metadata["scorecard_version"] == 1
    assert workspace.feedback() == [record]


def test_disagreeing_records_the_correct_label_and_the_comment(workspace):
    question = next_question(workspace, SCORE, random.Random(2))
    other = "negative" if question.result.value == "positive" else "positive"

    record = record_label(workspace, question, DISAGREE, correct_label=other.upper(),
                          comment="Sarcasm. It reads positive but is not.")

    assert record.final_answer_value == other            # normalized to the configured spelling
    assert record.initial_answer_value == question.result.value
    assert record.is_agreement is False
    assert "Sarcasm" in record.edit_comment_value


def test_disagreeing_needs_a_correct_label_that_is_one_of_the_classes(workspace):
    question = next_question(workspace, SCORE, random.Random(3))

    with pytest.raises(ValueError, match="correct label"):
        record_label(workspace, question, DISAGREE)
    with pytest.raises(ValueError, match="not one of"):
        record_label(workspace, question, DISAGREE, correct_label="maybe")
    assert workspace.feedback() == []                     # nothing half-written


def test_an_unknown_verdict_is_rejected(workspace):
    question = next_question(workspace, SCORE, random.Random(4))

    with pytest.raises(ValueError, match="unknown verdict"):
        record_label(workspace, question, "shrug")


def test_skipping_removes_the_item_from_selection_without_creating_a_trainable_label(workspace):
    question = next_question(workspace, SCORE, random.Random(5))

    record = record_label(workspace, question, SKIP)

    assert record.label is None
    assert workspace.n_labeled(SCORE) == 0
    assert question.item.id in workspace.labeled_ids(SCORE)


def test_too_few_labels_hold_the_incumbent_and_the_attempt_is_recorded(workspace):
    label_like_a_human(workspace, 10)

    outcome = refit(workspace, SCORE)

    assert outcome.status == "held"
    assert workspace.version == 1
    assert "more are needed" in outcome.reasons[0]
    event = workspace.events("fit")[-1]
    assert event["fitted"] is False and event["n_labeled"] == 10


def test_enough_labels_produce_a_better_calibrated_head_that_is_promoted(workspace):
    label_like_a_human(workspace, 90)

    outcome = refit(workspace, SCORE)

    assert outcome.promoted
    assert workspace.version == 2
    decision = workspace.scorecard().score(SCORE).decision
    assert decision.model == "multinomial_logistic"           # replaced the hand-written rule
    assert decision.calibration["method"] == "temperature"
    assert decision.provenance["tier"] == "shrunk"
    assert workspace.lineage()[-1]["kind"] == "fit"


def test_the_promoted_head_is_less_overconfident_than_jev_was(workspace):
    # The measured headline: Jev's raw confidence averages far above its accuracy.
    label_like_a_human(workspace, 90)
    before = [workspace.predict(i.id, SCORE).confidence for i in workspace.split("test")[:400]]
    refit(workspace, SCORE)

    after = [workspace.predict(i.id, SCORE).confidence for i in workspace.split("test")[:400]]

    assert sum(after) / len(after) < sum(before) / len(before) - 0.05


def test_refitting_on_the_same_labels_does_not_churn_versions(workspace):
    # The new incumbent was fit on these very labels, so a fresh out-of-fold candidate
    # cannot beat it. Promoting anyway would fill the lineage with noise.
    label_like_a_human(workspace, 90)
    refit(workspace, SCORE)

    second = refit(workspace, SCORE)

    assert second.status == "rejected"
    assert workspace.version == 2
    assert second.reasons


def test_a_dry_run_reports_what_would_happen_without_committing(workspace):
    label_like_a_human(workspace, 90)

    outcome = refit(workspace, SCORE, dry_run=True)

    assert outcome.promoted
    assert workspace.version == 1


def test_every_refit_attempt_is_recorded_whatever_its_outcome(workspace):
    label_like_a_human(workspace, 90)
    refit(workspace, SCORE)
    refit(workspace, SCORE)

    events = workspace.events("fit")

    assert [e["status"] for e in events] == ["promoted", "rejected"]
    assert events[0]["log_loss"] is not None and events[0]["new_version"] == 2


def test_status_reports_the_tier_and_how_far_the_next_one_is(workspace):
    label_like_a_human(workspace, 45)

    report = status(workspace, SCORE)

    assert report.n_labeled == 45
    assert report.tier == "shrunk"
    assert report.next_tier == "standard"
    assert report.distance_to_next == pytest.approx(200 - report.n_effective)
    assert set(report.triggers) == {"refit", "rethink"}


def test_status_says_a_refit_is_due_once_there_are_enough_new_labels(workspace):
    label_like_a_human(workspace, 45)

    assert status(workspace, SCORE).triggers["refit"].fire


def test_status_says_a_rethink_is_not_yet_worth_it_and_why(workspace):
    label_like_a_human(workspace, 45)

    rethink = status(workspace, SCORE).triggers["rethink"]

    assert not rethink.fire
    assert "not yet" in rethink.summary


def test_a_brand_new_workspace_reports_the_hold_tier(workspace):
    report = status(workspace, SCORE)

    assert report.tier == "hold" and report.n_labeled == 0
    assert not report.triggers["refit"].fire
