"""Feature: the committed demo recording reproduces the numbers the README quotes.

If one of these moves, the recording, the fixtures or the pipeline changed, and the README's
claims need re-checking before anything is published.
"""
from pathlib import Path

import pytest

pytest.importorskip("tactus")

from jev_flywheel.recording import replay  # noqa: E402
from jev_flywheel.report import complete_items, history  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / "fixtures" / "recordings" / "simulated-labeler"
FIXTURES = ROOT / "fixtures"


@pytest.fixture(scope="module")
def points(tmp_path_factory):
    workspace = replay(RECORDING, tmp_path_factory.mktemp("demo") / "var", FIXTURES)
    items = complete_items(workspace)
    return items, history(workspace, "Sentiment", item_ids=items)


def test_the_recording_scores_every_version_on_the_same_600_held_out_items(points):
    items, versions = points

    assert len(items) == 600
    assert [p.kind for p in versions] == ["seed"] + ["fit"] * (len(versions) - 2) + ["steer"]
    assert {p.scoreboard.summary.n for p in versions} == {600}


def test_jev_alone_is_accurate_enough_but_badly_overconfident(points):
    v1 = points[1][0].scoreboard.summary

    assert v1.accuracy == pytest.approx(0.768, abs=0.002)
    assert v1.ece == pytest.approx(0.151, abs=0.002)


def test_a_refit_fixes_calibration_without_changing_accuracy(points):
    versions = points[1]
    first = versions[0].scoreboard.summary
    last_refit = [p for p in versions if p.kind == "fit"][-1].scoreboard.summary

    assert last_refit.ece < first.ece / 2          # calibration is what a refit buys
    assert last_refit.accuracy == pytest.approx(first.accuracy, abs=0.01)


def test_the_steering_round_lifts_held_out_accuracy_and_keeps_calibration(points):
    versions = points[1]
    before = [p for p in versions if p.kind == "fit"][-1].scoreboard.summary
    after = versions[-1].scoreboard.summary

    assert versions[-1].kind == "steer"
    assert after.accuracy == pytest.approx(0.870, abs=0.002)
    assert after.accuracy > before.accuracy + 0.05      # steering is what buys accuracy
    assert after.ece < 0.05
    assert after.brier < before.brier
