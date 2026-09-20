"""The labeling console: one item at a time, agree or disagree.

This is the twenty-questions game. The system shows the item it would learn the most
from, says what it predicts and why, and asks whether you agree. A comment is optional
but is the most valuable thing you can give it: it is what the meta-cognition step reads
to work out which factor the scorecard is missing.

What the human sees, in order:

* the item's text;
* our answer and how confident we are (calibrated, with the raw figure beside it, so the
  effect of calibration is visible);
* what Jev alone said, when that differs;
* the few features that drove the decision, and the answer to every element question;
* *why this item was chosen* -- the selection components -- so the choice is legible and
  not a black box.

Input is injected (``read_key``, ``read_line``) so the whole console can be driven by
tests, and rendering is a pure function so what is shown is spec'd like everything else.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rich.console import Console as RichConsole
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jev_flywheel.loop import (
    AGREE, DISAGREE, SKIP, Question, RefitOutcome, Status, describe_answer, next_question,
    record_label, refit, status)
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.selection import SelectionPolicy
from jev_flywheel.workspace import Workspace

KEYS = {"a": AGREE, "d": DISAGREE, "s": SKIP, "q": "quit"}


def confidence_line(question: Question) -> str:
    result = question.result
    line = f"{result.value}  ({(result.confidence or 0):.0%} confident"
    raw = result.raw_confidence
    if raw is not None and result.confidence is not None and abs(raw - result.confidence) > 0.005:
        line += f", raw {raw:.0%}"
    return line + ")"


def render_question(question: Question, card: Scorecard, number: int, state: Optional[Status] = None):
    """The whole screen for one item, as a rich renderable."""
    score = card.score(question.result.score_name)
    result = question.result

    title = f"Question {number} · scorecard v{question.scorecard_version} · {score.name}"
    if state is not None:
        title += f" · {state.n_labeled} labeled · tier {state.tier}"
    body = Panel(Text(question.item.text), title=title, title_align="left", padding=(1, 2))

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("We say", confidence_line(question))
    jev = result.metadata.get("jev") or {}
    if jev.get("value") is not None and jev["value"] != result.value:
        summary.add_row("Jev alone", f"{jev['value']}  ({jev['confidence']:.0%})  "
                                     "[yellow]- we overruled it[/yellow]")
    elif jev.get("value") is not None:
        summary.add_row("Jev alone", f"{jev['value']}  ({jev['confidence']:.0%})  (agrees)")
    drivers = (result.metadata.get("decision") or {}).get("top_contributions") or []
    if drivers:
        summary.add_row("Why", ", ".join(f"{d['feature']} ({d['contribution']:+.2f})" for d in drivers))

    parts = [body, summary]
    elements = [(wire, spec) for _, wire, spec in score.element_questions()]
    if elements:
        table = Table(title="Element answers", title_justify="left", box=None, show_header=False)
        table.add_column(style="dim")
        table.add_column()
        for wire, spec in elements:
            answer = question.answers.get(wire)
            table.add_row(spec.key, describe_answer(answer) if answer else "[dim]no answer[/dim]")
        parts.append(table)

    record = question.selection.record()
    components = ", ".join(f"{k} {v:.2f}" for k, v in record["selection_components"].items())
    parts.append(Text(
        f"Chosen because: {components}  (picked with probability {record['propensity']:.4f} "
        f"from {record['pool_size']} unlabeled)", style="dim"))
    parts.append(Text("[a]gree   [d]isagree   [s]kip   [q]uit", style="bold cyan"))
    return Group(*parts)


@dataclass
class Session:
    """What a console run did, for the caller and for tests."""

    labeled: int = 0
    agreed: int = 0
    skipped: int = 0
    refits: List[RefitOutcome] = field(default_factory=list)
    rethink_offered: int = 0
    stopped: bool = False


def default_read_key() -> str:
    import click

    return click.getchar().lower()


def default_read_line(prompt: str) -> str:
    return input(prompt)


def run(
    workspace: Workspace,
    score_name: str,
    *,
    rng: Optional[random.Random] = None,
    editor: str = "human",
    max_questions: Optional[int] = None,
    console: Optional[RichConsole] = None,
    read_key: Callable[[], str] = default_read_key,
    read_line: Callable[[str], str] = default_read_line,
    policy: SelectionPolicy = SelectionPolicy(),
    auto_refit: bool = True,
    on_rethink: Optional[Callable[[Workspace, str], None]] = None,
) -> Session:
    """Ask questions until the human quits or there is nothing left to ask.

    After every label the steering policy is consulted. A refit is cheap and runs by
    itself, and only promotes if it beats the incumbent out of fold. A rethink costs an
    LLM call and possibly a Jev top-up, so it is offered, not forced: ``on_rethink`` runs
    it if given, and otherwise the console says it is worth doing.
    """
    console = console or RichConsole()
    rng = rng or random.Random()
    session = Session()
    asked = 0

    while max_questions is None or asked < max_questions:
        question = next_question(workspace, score_name, rng, policy)
        if question is None:
            console.print("[green]Every item in the pool has been asked about.[/green]")
            break
        asked += 1
        card = workspace.scorecard()
        console.print()
        console.print(render_question(question, card, asked, status(workspace, score_name)))

        verdict = _ask_verdict(read_key)
        if verdict == "quit":
            session.stopped = True
            break
        correct, comment = None, None
        if verdict == DISAGREE:
            correct = _ask_correct_label(question, read_line, console)
        if verdict in (AGREE, DISAGREE):
            comment = read_line("Why? (optional, enter to skip) > ").strip() or None
        record_label(workspace, question, verdict, correct_label=correct,
                     comment=comment, editor=editor)
        if verdict == SKIP:
            session.skipped += 1
            console.print("[dim]Skipped.[/dim]")
            continue
        session.labeled += 1
        session.agreed += verdict == AGREE
        console.print(f"[dim]Recorded ({'agree' if verdict == AGREE else 'disagree'}"
                      f"{', with a comment' if comment else ''}).[/dim]")

        state = status(workspace, score_name)
        if auto_refit and state.triggers["refit"].fire:
            outcome = refit(workspace, score_name)
            session.refits.append(outcome)
            console.print(_describe_refit(outcome))
            state = status(workspace, score_name)
        if state.triggers["rethink"].fire:
            session.rethink_offered += 1
            console.print(f"[bold magenta]{state.triggers['rethink'].summary}[/bold magenta]")
            if on_rethink is not None:
                on_rethink(workspace, score_name)
    return session


def _ask_verdict(read_key: Callable[[], str]) -> str:
    while True:
        key = read_key()
        if key in KEYS:
            return KEYS[key]


def _ask_correct_label(question: Question, read_line, console: RichConsole) -> str:
    classes = question.classes
    others = [c for c in classes if c != question.result.value]
    if len(others) == 1:
        return others[0]                     # two classes: disagreeing names the other one
    listing = "  ".join(f"{i + 1}={c}" for i, c in enumerate(classes))
    while True:
        answer = read_line(f"Correct label ({listing}) > ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(classes):
            return classes[int(answer) - 1]
        if answer in classes:
            return answer
        console.print("[red]Not one of the labels.[/red]")


def _describe_refit(outcome: RefitOutcome) -> str:
    if outcome.promoted:
        result = outcome.result
        return (f"[bold green]Refit promoted: scorecard v{outcome.version}[/bold green] "
                f"(tier {result.tier.name}, out-of-fold accuracy {result.metrics.accuracy:.1%}, "
                f"ECE {result.metrics.ece:.3f})")
    label = {"held": "Refit held", "rejected": "Refit rejected", "refused": "Refit refused"}[
        outcome.status]
    return f"[dim]{label}: {outcome.reasons[0] if outcome.reasons else ''}[/dim]"
