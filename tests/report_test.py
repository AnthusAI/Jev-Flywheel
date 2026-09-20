"""Feature: measuring the flywheel honestly.

The scoreboard and the alignment curve answer different questions and must not be
confused. The specs run on a slice of the real corpus.
"""
import random
import shutil

import pytest

from jev_flywheel.items import FeedbackItem, normalize_label
from jev_flywheel.loop import AGREE, DISAGREE, next_question, record_label, refit
from jev_flywheel.report import alignment_curve, confusion, history, scoreboard
from jev_flywheel.workspace import Workspace
from tests.loop_test import SCORE, miniature


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("template")
    return Workspace.init(root / "var", miniature(root / "fixtures"))


@pytest.fixture
def workspace(template, tmp_path):
    shutil.copytree(template.root, tmp_path / "var")
    return Workspace(tmp_path / "var")


def label(workspace, count, seed=0):
    rng = random.Random(seed)
    for _ in range(count):
        question = next_question(workspace, SCORE, rng)
        truth = question.item.reference_label
        if normalize_label(truth) == normalize_label(question.result.value):
            record_label(workspace, question, AGREE)
        else:
            record_label(workspace, question, DISAGREE, correct_label=truth)


def test_the_scoreboard_reports_accuracy_on_the_held_out_test_split(workspace):
    board = scoreboard(workspace, SCORE)

    assert board.split == "test"
    assert board.summary.n == 200
    assert 0.65 < board.accuracy < 0.85            # Jev's holistic answer, ~0.76 on the full split
    assert board.version == 1


def test_the_scoreboard_breaks_accuracy_down_by_tier(workspace):
    board = scoreboard(workspace, SCORE)

    assert set(board.by_tier) == {"strong", "medium", "weak", "neutral"}
    assert board.by_tier["strong"] > board.by_tier["neutral"]


def test_jev_alone_is_overconfident_on_the_scoreboard(workspace):
    # The headline failure this whole project exists to correct.
    assert scoreboard(workspace, SCORE).summary.overconfidence > 0.05


def test_the_scoreboard_never_reads_anything_the_human_labeled(workspace):
    label(workspace, 30)
    labeled = workspace.labeled_ids(SCORE)

    test_ids = {i.id for i in workspace.split("test")}

    assert labeled.isdisjoint(test_ids)


def test_a_fitted_head_improves_calibration_on_the_scoreboard(workspace):
    before = scoreboard(workspace, SCORE)
    label(workspace, 90)
    assert refit(workspace, SCORE).promoted

    after = scoreboard(workspace, SCORE)

    assert after.version == 2
    assert after.summary.ece < before.summary.ece
    assert after.summary.brier < before.summary.brier


def test_an_older_version_can_still_be_scored_after_a_newer_one_exists(workspace):
    label(workspace, 90)
    refit(workspace, SCORE)

    v1 = scoreboard(workspace, SCORE, version=1)

    assert v1.version == 1
    assert v1.summary.overconfidence > 0.05


def test_the_reliability_bins_are_available_for_a_diagram(workspace):
    bins = scoreboard(workspace, SCORE).bins

    assert bins and all(0 <= b.accuracy <= 1 for b in bins)


def test_history_places_each_version_by_how_much_feedback_existed_when_it_was_made(workspace):
    label(workspace, 90)
    refit(workspace, SCORE)

    points = history(workspace, SCORE)

    assert [(p.version, p.kind) for p in points] == [(1, "seed"), (2, "fit")]
    assert points[0].n_feedback == 0
    assert points[1].n_feedback == 90
    assert points[1].scoreboard.summary.ece < points[0].scoreboard.summary.ece


def test_the_alignment_curve_has_one_point_per_label_and_is_a_valid_rate(workspace):
    label(workspace, 25)

    curve = alignment_curve(workspace, SCORE)

    assert [p.k for p in curve] == list(range(1, 26))
    assert all(0.0 <= p.agreement <= 1.0 for p in curve)
    assert curve[-1].n_effective <= 25


def test_alignment_is_measured_on_the_prediction_shown_before_the_label_existed(workspace):
    # Prequential: each agree/disagree is an out-of-sample trial, so no split is needed.
    label(workspace, 25)

    shown = [f for f in workspace.feedback() if f.is_agreement is not None]

    assert all(f.initial_answer_value is not None for f in shown)
    assert all(f.metadata["scorecard_version"] == 1 for f in shown)


def test_a_heavily_weighted_item_moves_the_alignment_more_than_a_light_one(workspace):
    def fb(idx, agreed, propensity):
        return FeedbackItem(
            id=f"f{idx}", item_id=f"x{idx}", score_name=SCORE, initial_answer_value="positive",
            final_answer_value="positive" if agreed else "negative", is_agreement=agreed,
            edited_at=f"2026-01-01T00:00:0{idx}", metadata={"propensity": propensity})
    for record in (fb(1, True, 0.5), fb(2, True, 0.5), fb(3, False, 0.005)):
        workspace.add_feedback(record)

    final = alignment_curve(workspace, SCORE)[-1]

    # The disagreement was 100x less likely to be shown, so it stands in for ~100x more of
    # the pool. Unweighted this would read 2/3; weighted it must read far lower.
    assert final.agreement < 0.5


def test_a_confusion_matrix_is_reported_against_the_reference_labels(workspace):
    labels, matrix = confusion(workspace, SCORE)

    assert labels == ["negative", "positive"]
    assert sum(sum(row) for row in matrix) == 200
