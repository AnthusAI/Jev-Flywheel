"""Feature: a recorded flywheel run replays offline and reproduces the same lineage."""
import json
import random
import shutil

import pytest

from jev_flywheel.recording import RecordingError, load, record, replay
from jev_flywheel.report import complete_items, history
from jev_flywheel.steer import ScriptedApprover, run_steering
from jev_flywheel.workspace import Workspace
from tests.host_test import SARCASM, Client, label, reply
from tests.loop_test import SCORE, miniature

pytest.importorskip("tactus")


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return miniature(tmp_path_factory.mktemp("rec") / "fixtures", n_pool=700, n_test=200)


@pytest.fixture(scope="module")
def session(fixtures, tmp_path_factory):
    """A real session: labels, a promoted refit, then one promoted steering round."""
    from jev_flywheel.loop import refit

    root = tmp_path_factory.mktemp("session")
    workspace = Workspace.init(root / "var", fixtures)
    label(workspace, 100)
    assert refit(workspace, SCORE).promoted
    label(workspace, 40, seed=1)
    client = Client(workspace)
    outcome = run_steering(
        workspace, SCORE, allow_spend=True, client_factory=lambda: client,
        hitl_handler=ScriptedApprover([True]), mock_replies=[reply(add_elements=[SARCASM])])
    assert outcome.promoted
    # Score the steered version on some held-out items, as a live run would have.
    workspace.cache  # noqa: B018
    import asyncio
    questions = workspace.scorecard().questions()
    from jev_flywheel.jev import JevSession
    asyncio.run(workspace.cache.fill(
        JevSession(client_factory=lambda: client), workspace.split("test")[:80], questions))
    recording = record(workspace, SCORE, root / "recording", fixtures,
                       title="Test session", provenance="Simulated labeler.")
    return workspace, recording


def test_a_recording_captures_labels_answers_and_every_step(session):
    workspace, recording = session
    script = load(recording)

    assert script["n_labels"] == 140
    assert [s["op"] for s in script["steps"]].count("steer") == 1
    steer = next(s for s in script["steps"] if s["op"] == "steer")
    assert json.loads(steer["analyst_reply"])["add_elements"][0]["key"] == "sarcasm"
    assert steer["approve"] is True
    assert (recording / "feedback.jsonl").exists()
    assert (recording / "extra_answers.jsonl.gz").exists()
    assert "Simulated labeler." in (recording / "README.md").read_text()


def test_the_recording_holds_only_what_the_fixtures_do_not(session):
    _, recording = session
    import gzip
    rows = [json.loads(line) for line in gzip.open(recording / "extra_answers.jsonl.gz", "rt")]

    # Only answers to the proposed element: everything else is in the bundled fixtures.
    assert rows and {r["name"] for r in rows} == {"sentiment.sarcasm"}


def test_replaying_reproduces_the_same_scorecard_lineage(session, fixtures, tmp_path):
    original, recording = session

    replayed = replay(recording, tmp_path / "replayed", fixtures)

    assert [(e["version"], e["kind"]) for e in replayed.lineage()] == \
           [(e["version"], e["kind"]) for e in original.lineage()]
    assert replayed.n_labeled(SCORE) == original.n_labeled(SCORE)


def test_replaying_reproduces_the_same_fitted_numbers(session, fixtures, tmp_path):
    original, recording = session

    replayed = replay(recording, tmp_path / "replayed", fixtures)

    a = original.scorecard().score(SCORE).decision
    b = replayed.scorecard().score(SCORE).decision
    assert a.provenance["fit_id"] == b.provenance["fit_id"]
    for feature, weight in a.weights["positive"].items():
        assert b.weights["positive"][feature] == pytest.approx(weight, abs=1e-9)


def test_replaying_reproduces_the_same_held_out_scoreboard(session, fixtures, tmp_path):
    original, recording = session
    replayed = replay(recording, tmp_path / "replayed", fixtures)
    items = complete_items(original)

    before = [p.scoreboard.summary for p in history(original, SCORE, item_ids=items)]
    after = [p.scoreboard.summary for p in history(replayed, SCORE, item_ids=items)]

    assert [round(s.accuracy, 6) for s in before] == [round(s.accuracy, 6) for s in after]
    assert [round(s.ece, 6) for s in before] == [round(s.ece, 6) for s in after]


def test_a_replay_reports_each_step_as_it_happens(session, fixtures, tmp_path):
    _, recording = session
    seen = []

    replay(recording, tmp_path / "replayed", fixtures, on_step=seen.append)

    assert any("refit promoted" in m for m in seen)
    assert any("steering promoted" in m for m in seen)


def test_a_replay_never_touches_the_network_or_a_model(session, fixtures, tmp_path, monkeypatch):
    import litellm

    def boom(*a, **k):
        raise AssertionError("a replay reached out to a model")

    monkeypatch.setattr(litellm, "completion", boom)
    monkeypatch.setattr(litellm, "acompletion", boom)
    _, recording = session

    replay(recording, tmp_path / "replayed", fixtures)      # no Jev client is even provided


def test_a_directory_that_is_not_a_recording_is_refused(tmp_path):
    with pytest.raises(RecordingError, match="script.json"):
        load(tmp_path)


def test_the_figure_is_written_from_the_replayed_session(session, fixtures, tmp_path):
    pytest.importorskip("matplotlib")
    from jev_flywheel.charts import save_chart

    _, recording = session
    replayed = replay(recording, tmp_path / "replayed", fixtures)

    png = save_chart(replayed, SCORE, tmp_path / "results.png")

    assert png.exists() and png.stat().st_size > 10_000


def test_the_cli_replays_a_recording_and_prints_the_lineage(session, fixtures, tmp_path):
    from click.testing import CliRunner
    from jev_flywheel.cli import cli

    _, recording = session

    result = CliRunner().invoke(cli, ["--workspace", str(tmp_path / "w"), "replay",
                                     str(recording), "--fixtures", str(fixtures)])

    assert result.exit_code == 0, result.output
    assert "Test session" in result.output
    assert "v1" in result.output and "v3" in result.output and "steer" in result.output
