"""Feature: the flywheel command line, run through click's test runner, offline."""
import random

import pytest
from click.testing import CliRunner

from jev_flywheel.cli import cli
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.workspace import Workspace
from tests.loop_test import SCORE, label_like_a_human, miniature


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return miniature(tmp_path_factory.mktemp("cli") / "fixtures")


@pytest.fixture
def home(tmp_path, fixtures):
    return tmp_path / "var"


@pytest.fixture
def ready(home, fixtures):
    Workspace.init(home, fixtures)
    return Workspace(home)


def run(home, *args, obj=None, input=None):
    return CliRunner().invoke(cli, ["--workspace", str(home), *args], obj=obj, input=input)


def test_init_builds_a_workspace_and_says_how_many_items_are_held_out(home, fixtures):
    result = run(home, "init", "--fixtures", str(fixtures))

    assert result.exit_code == 0, result.output
    assert "600 items to ask about" in result.output
    assert "200 held out" in result.output
    assert "flywheel label" in result.output


def test_init_refuses_to_clobber_a_workspace_without_force(ready, home, fixtures):
    result = run(home, "init", "--fixtures", str(fixtures))

    assert result.exit_code != 0
    assert "--force" in result.output


def test_a_command_without_a_workspace_says_to_run_init(tmp_path):
    result = run(tmp_path / "nowhere", "status")

    assert result.exit_code != 0
    assert "flywheel init" in result.output


def test_status_shows_the_tier_and_both_triggers_with_their_conditions(ready, home):
    label_like_a_human(ready, 45)

    result = run(home, "status")

    assert result.exit_code == 0, result.output
    assert "45 labeled" in result.output
    assert "tier shrunk" in result.output
    assert "refit: WORTH DOING NOW" in result.output
    assert "rethink: not yet" in result.output
    # Collapse whitespace: the table wraps long condition names at 80 columns.
    assert "mismatches the human explained" in " ".join(result.output.split())


def test_fit_holds_when_there_are_too_few_labels_and_says_how_many_more(ready, home):
    label_like_a_human(ready, 10)

    result = run(home, "fit")

    assert result.exit_code == 0, result.output
    assert "HELD" in result.output
    assert "more are needed" in result.output


def test_fit_promotes_a_new_scorecard_version_and_reports_out_of_fold_numbers(ready, home):
    label_like_a_human(ready, 90)

    result = run(home, "fit")

    assert "PROMOTED -> scorecard v2" in result.output
    assert "out-of-fold" in result.output and "incumbent" in result.output
    assert Workspace(home).version == 2


def test_a_dry_run_fit_commits_nothing(ready, home):
    label_like_a_human(ready, 90)

    result = run(home, "fit", "--dry-run")

    assert "PROMOTED" in result.output
    assert Workspace(home).version == 1


def test_evaluate_reports_the_held_out_scoreboard_and_alignment(ready, home):
    label_like_a_human(ready, 30)

    result = run(home, "evaluate")

    assert result.exit_code == 0, result.output
    assert "200 held-out items no human has seen" in result.output
    assert "By tier:" in result.output
    assert "Alignment" in result.output and "30 labels" in result.output


def test_evaluate_with_no_labels_says_alignment_is_not_available_yet(ready, home):
    result = run(home, "evaluate")

    assert "No labels yet" in result.output


def test_history_lists_every_version_with_its_scoreboard(ready, home):
    label_like_a_human(ready, 90)
    run(home, "fit")

    result = run(home, "history")

    assert result.exit_code == 0, result.output
    assert "v1" in result.output and "v2" in result.output
    assert "seed" in result.output and "fit" in result.output


def test_label_quits_cleanly_on_q(ready, home):
    result = run(home, "label", "--seed", "1", input="q")

    assert result.exit_code == 0, result.output
    assert "0 labeled" in result.output


def test_a_score_flag_naming_a_missing_score_is_rejected(ready, home):
    result = run(home, "status", "--score", "Nonsense")

    assert result.exit_code != 0
    assert "Nonsense" in result.output


# ---- topup: prices first, spends only on --yes ------------------------------------

class Usage:
    def model_dump(self):
        return {"input_tokens": 100, "output_tokens": 5}


class Response:
    model = "jev-test"
    usage = Usage()

    def __init__(self, answers):
        self.answers = answers


class Client:
    def __init__(self):
        self.calls = []

    async def system_one(self, *, state, questions):
        self.calls.append(sorted(questions))
        return Response({name: {"type": "noul", "noul": 0.5} for name in questions})


def add_an_element(workspace):
    config = workspace.scorecard().to_config()
    config["scores"][0]["elements"] = [
        {"key": "sarcasm", "question_type": "noul", "instructions": "Is this sarcastic?"}]
    workspace.commit_scorecard(Scorecard.from_config(config), kind="steer")


def test_topup_with_nothing_missing_says_so(ready, home):
    result = run(home, "topup", "--items", "pool")

    assert "Nothing to fetch" in result.output


def test_topup_prices_the_spend_and_sends_nothing_without_yes(ready, home):
    add_an_element(ready)
    client = Client()

    result = run(home, "topup", "--items", "pool", obj={"client_factory": lambda: client})

    assert "600 need a Jev request" in result.output
    assert "Not spending anything" in result.output
    assert client.calls == []


def test_topup_with_yes_asks_only_for_the_missing_question_once_per_item(ready, home):
    add_an_element(ready)
    client = Client()
    label_like_a_human(ready, 3)     # a few labeled items, so 'labeled' has something to do

    result = run(home, "topup", "--items", "labeled", "--yes", obj={"client_factory": lambda: client})

    assert result.exit_code == 0, result.output
    assert len(client.calls) == 3
    assert all(call == ["sentiment.sarcasm"] for call in client.calls)
    assert "3 requests sent, 0 failed" in result.output
    # ...and afterwards there is nothing left to fetch for those items.
    assert "Nothing to fetch" in run(home, "topup", "--items", "labeled").output


# ---- steer -------------------------------------------------------------------------

def _reply_file(tmp_path, **kwargs):
    import json

    path = tmp_path / "reply.json"
    path.write_text(json.dumps({"root_cause": "sarcasm goes undetected", **kwargs}))
    return path


SARCASM_ELEMENT = {"key": "sarcasm", "question_type": "noul",
                   "instructions": "Is the writer being sarcastic, so the words say the opposite?"}


@pytest.fixture
def labeled_home(ready, home):
    label_like_a_human(ready, 100, comment="reads positive but it is sarcasm")
    return ready


def steer_obj(workspace, approvals=(True,)):
    from jev_flywheel.steer import ScriptedApprover
    from tests.host_test import Client

    return {"client_factory": lambda: Client(workspace), "hitl_handler": ScriptedApprover(approvals)}


def test_steer_runs_a_round_and_reports_the_decision_and_the_new_version(labeled_home, home, tmp_path):
    reply = _reply_file(tmp_path, add_elements=[SARCASM_ELEMENT])

    result = run(home, "steer", "--allow-spend", "--scripted-reply", str(reply),
                 obj=steer_obj(labeled_home))

    assert result.exit_code == 0, result.output
    assert "Decision: promoted" in result.output
    assert "sarcasm goes undetected" in result.output
    assert "is now the current version" in result.output
    assert Workspace(home).version == 2


def test_steer_stops_at_the_price_without_allow_spend(labeled_home, home, tmp_path):
    reply = _reply_file(tmp_path, add_elements=[SARCASM_ELEMENT])

    result = run(home, "steer", "--scripted-reply", str(reply), obj=steer_obj(labeled_home))

    assert "Decision: needs_spend" in result.output
    assert "--allow-spend" in result.output
    assert Workspace(home).version == 1


def test_steer_with_a_declined_approval_changes_nothing(labeled_home, home, tmp_path):
    reply = _reply_file(tmp_path, add_elements=[SARCASM_ELEMENT])

    result = run(home, "steer", "--allow-spend", "--scripted-reply", str(reply),
                 obj=steer_obj(labeled_home, approvals=(False,)))

    assert "Decision: rejected_by_human" in result.output
    assert Workspace(home).version == 1


def test_steer_reports_a_proposal_of_no_change(labeled_home, home, tmp_path):
    result = run(home, "steer", "--scripted-reply", str(_reply_file(tmp_path)),
                 obj=steer_obj(labeled_home))

    assert "Decision: no_change_proposed" in result.output


def test_steer_is_listed_beside_the_other_commands(ready, home):
    assert "steer" in run(home, "--help").output
