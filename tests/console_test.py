"""Feature: the labeling console, driven by scripted keystrokes."""
import io
import random
import shutil

import pytest
from rich.console import Console as RichConsole

from jev_flywheel.console import render_question, run
from jev_flywheel.loop import next_question
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


def screen():
    return RichConsole(file=io.StringIO(), record=True, width=110, force_terminal=False)


def text_of(console):
    # Collapse whitespace so a sentence the terminal wrapped still matches.
    return " ".join(console.export_text().split())


def keys(*presses):
    it = iter(presses)
    return lambda: next(it)


def lines(*answers):
    it = iter(answers)
    return lambda prompt: next(it, "")


def render(workspace, seed=0):
    question = next_question(workspace, SCORE, random.Random(seed))
    console = screen()
    console.print(render_question(question, workspace.scorecard(), 1))
    return question, text_of(console)


def test_the_screen_shows_the_item_text_our_answer_and_how_confident_we_are(workspace):
    question, output = render(workspace)

    assert question.item.text[:40] in output
    assert "We say" in output
    assert question.result.value in output
    assert "confident" in output


def test_the_screen_says_why_this_item_was_chosen_and_how_likely_it_was_to_be_picked(workspace):
    _, output = render(workspace)

    assert "Chosen because" in output
    assert "uncertainty" in output and "novelty" in output
    assert "picked with probability" in output


def test_the_screen_shows_what_jev_alone_said(workspace):
    _, output = render(workspace)

    assert "Jev alone" in output


def test_a_calibrated_confidence_is_shown_beside_the_raw_one_so_the_effect_is_visible(workspace):
    from jev_flywheel.console import confidence_line
    from jev_flywheel.loop import refit
    from tests.loop_test import label_like_a_human

    label_like_a_human(workspace, 90)
    assert refit(workspace, SCORE).promoted

    question = next_question(workspace, SCORE, random.Random(1))

    assert "raw" in confidence_line(question)


def test_the_screen_lists_the_answer_to_every_element_question(workspace):
    from jev_flywheel.loop import Question
    from jev_flywheel.scorecard import Scorecard
    from tests.loop_test import FIXTURES

    reference = Scorecard.from_yaml((FIXTURES / "scorecards" / "reference_full.yaml").read_text())
    question = next_question(workspace, SCORE, random.Random(0))
    with_elements = Question(
        question.item, question.result, question.selection, question.scorecard_version,
        workspace.cache.partial_answers_for(question.item.id, reference.questions()),
        question.classes)
    console = screen()

    console.print(render_question(with_elements, reference, 1))

    output = text_of(console)
    assert "Element answers" in output
    for element in ("praise", "criticism", "irony", "expectation", "intensity"):
        assert element in output


def test_agreeing_writes_an_agreement_and_disagreeing_names_the_other_class(workspace):
    console = screen()

    session = run(workspace, SCORE, rng=random.Random(0), console=console, editor="ryan",
                  read_key=keys("a", "d", "q"), read_line=lines("clear positive", "actually sarcasm"),
                  auto_refit=False)

    records = workspace.feedback()
    assert [r.is_agreement for r in records] == [True, False]
    assert records[0].edit_comment_value == "clear positive"
    assert records[1].edit_comment_value == "actually sarcasm"
    assert records[1].final_answer_value != records[1].initial_answer_value
    assert records[0].editor_name == "ryan"
    assert (session.labeled, session.agreed, session.stopped) == (2, 1, True)


def test_a_blank_comment_is_stored_as_no_comment(workspace):
    run(workspace, SCORE, rng=random.Random(0), console=screen(), read_key=keys("a", "q"),
        read_line=lines(""), auto_refit=False)

    assert workspace.feedback()[0].edit_comment_value is None


def test_skipping_asks_for_no_comment_and_records_no_label(workspace):
    session = run(workspace, SCORE, rng=random.Random(0), console=screen(),
                  read_key=keys("s", "q"), read_line=lines("should never be read"),
                  auto_refit=False)

    assert session.skipped == 1 and session.labeled == 0
    assert workspace.n_labeled(SCORE) == 0
    assert workspace.feedback()[0].edit_comment_value is None


def test_unrecognized_keys_are_ignored_until_a_real_one_arrives(workspace):
    session = run(workspace, SCORE, rng=random.Random(0), console=screen(),
                  read_key=keys("x", "9", " ", "a", "q"), read_line=lines(""), auto_refit=False)

    assert session.labeled == 1


def test_quitting_stops_immediately_and_writes_nothing(workspace):
    session = run(workspace, SCORE, rng=random.Random(0), console=screen(),
                  read_key=keys("q"), read_line=lines(), auto_refit=False)

    assert session.stopped and workspace.feedback() == []


def test_the_question_limit_ends_the_session(workspace):
    session = run(workspace, SCORE, rng=random.Random(0), console=screen(), max_questions=3,
                  read_key=keys(*"aaa"), read_line=lines("", "", ""), auto_refit=False)

    assert session.labeled == 3 and not session.stopped


def test_a_refit_runs_by_itself_once_the_steering_policy_says_it_is_due(workspace):
    console = screen()

    session = run(workspace, SCORE, rng=random.Random(0), console=console, max_questions=45,
                  read_key=lambda: "a", read_line=lambda prompt: "")

    assert session.refits
    assert "Refit" in text_of(console)


def test_an_exhausted_pool_ends_the_session_and_says_so(tmp_path):
    root = tmp_path
    small = Workspace.init(root / "var", miniature(root / "fixtures", n_pool=3, n_test=5))
    console = screen()

    session = run(small, SCORE, rng=random.Random(0), console=console, read_key=lambda: "a",
                  read_line=lambda prompt: "", auto_refit=False)

    assert session.labeled == 3
    assert "Every item in the pool" in text_of(console)
