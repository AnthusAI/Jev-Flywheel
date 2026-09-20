"""The ``flywheel`` command line.

    flywheel init        create a workspace from the committed fixtures (offline)
    flywheel label       the console: active selection, agree or disagree, comments
    flywheel fit         refit the head from the labels so far; promote only if better
    flywheel status      which triggers have fired, and how far off the others are
    flywheel evaluate    accuracy on the held-out test split, and alignment with the human
    flywheel history     the scorecard's version lineage
    flywheel topup       ask Jev for answers the current scorecard is missing (prices it first)
    flywheel steer       the Tactus meta-cognition procedure
    flywheel replay      replay the recorded flywheel from the fixtures

Nothing here needs the network except ``topup`` and ``steer``, and both say what they
will spend before spending it.
"""
from __future__ import annotations

import asyncio
import getpass
import os
import random
from pathlib import Path
from typing import Optional

import click
from rich.console import Console as RichConsole
from rich.table import Table

from jev_flywheel import console as label_console
from jev_flywheel.answers import AnswerCache
from jev_flywheel.jev import JevSession
from jev_flywheel.loop import refit, status as loop_status
from jev_flywheel.report import alignment_curve, history as version_history, scoreboard
from jev_flywheel.workspace import Workspace, WorkspaceError

DEFAULT_WORKSPACE = "var"
PACKAGED_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _workspace(ctx: click.Context) -> Workspace:
    try:
        return Workspace(ctx.obj["workspace"]).require()
    except WorkspaceError as error:
        raise click.ClickException(str(error))


def _score_name(workspace: Workspace, requested: Optional[str]) -> str:
    """The score to operate on. A one-score card needs no flag; a bigger one does."""
    names = [s.name for s in workspace.scorecard().scores]
    if requested:
        if requested not in names:
            raise click.UsageError(f"no score {requested!r}; the scorecard has {names}")
        return requested
    if len(names) == 1:
        return names[0]
    raise click.UsageError(f"the scorecard has several scores {names}; pass --score")


score_option = click.option("--score", "score_name", default=None,
                            help="Which score. Optional when the scorecard has only one.")


class Group(click.Group):
    """Keeps the commands in the order they are written, which is the order of the loop."""

    def list_commands(self, ctx):
        return list(self.commands)


@click.group(cls=Group)
@click.option("--workspace", "-w", default=lambda: os.environ.get(
    "JEV_FLYWHEEL_WORKSPACE", DEFAULT_WORKSPACE), show_default="var", type=click.Path(path_type=Path),
    help="Where the flywheel keeps its state.")
@click.pass_context
def cli(ctx, workspace):
    """Jev plus a decision head that learns from your feedback."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace


@cli.command()
@click.option("--fixtures", type=click.Path(path_type=Path, exists=True), default=PACKAGED_FIXTURES,
              show_default="fixtures/")
@click.option("--force", is_flag=True, help="Replace an existing workspace. Deletes its labels.")
@click.pass_context
def init(ctx, fixtures, force):
    """Create a workspace from the committed fixtures. No network, no keys."""
    try:
        workspace = Workspace.init(ctx.obj["workspace"], fixtures, force=force)
    except WorkspaceError as error:
        raise click.ClickException(str(error))
    pool, test = len(workspace.split("pool")), len(workspace.split("test"))
    click.echo(f"Workspace ready at {workspace.root}: {pool} items to ask about, {test} held out "
               f"as the scoreboard, scorecard v{workspace.version}.")
    click.echo("Next: flywheel label")


@cli.command()
@score_option
@click.option("--count", "-n", type=int, default=None, help="Stop after this many questions.")
@click.option("--editor", default=lambda: getpass.getuser(), show_default="you",
              help="Who is labeling; recorded on every judgement.")
@click.option("--seed", type=int, default=None, help="Fix the random choices, for reproducibility.")
@click.option("--no-refit", is_flag=True, help="Do not refit automatically when one is due.")
@click.pass_context
def label(ctx, score_name, count, editor, seed, no_refit):
    """Answer questions one at a time: agree, disagree, or skip, with an optional comment."""
    workspace = _workspace(ctx)
    score_name = _score_name(workspace, score_name)
    session = label_console.run(
        workspace, score_name, rng=random.Random(seed), editor=editor, max_questions=count,
        auto_refit=not no_refit)
    click.echo(f"\n{session.labeled} labeled ({session.agreed} agreed), {session.skipped} skipped, "
               f"{len(session.refits)} refit(s) attempted.")


@cli.command()
@score_option
@click.option("--dry-run", is_flag=True, help="Report what would happen without committing.")
@click.pass_context
def fit(ctx, score_name, dry_run):
    """Refit the head from the labels so far. Promotes only if it beats the incumbent."""
    workspace = _workspace(ctx)
    score_name = _score_name(workspace, score_name)
    outcome = refit(workspace, score_name, dry_run=dry_run)
    result = outcome.result
    click.echo(f"{outcome.status.upper()}" + (f" -> scorecard v{outcome.version}"
                                              if outcome.version else ""))
    for reason in outcome.reasons:
        click.echo(f"  {reason}")
    if result and result.fitted:
        click.echo(f"  tier {result.tier.name}, {result.n} labels "
                   f"({result.n_effective:.1f} effective), C={result.chosen_c}, "
                   f"calibration {result.calibration['method']}")
        click.echo(f"  out-of-fold: accuracy {result.metrics.accuracy:.3f}, "
                   f"ECE {result.metrics.ece:.3f}, Brier {result.metrics.brier:.3f}")
        if outcome.comparison is not None:
            incumbent = outcome.comparison.incumbent
            click.echo(f"  incumbent:   accuracy {incumbent.accuracy:.3f}, "
                       f"ECE {incumbent.ece:.3f}, Brier {incumbent.brier:.3f}")
    if outcome.needs_answers:
        click.echo(f"  {outcome.needs_answers} labeled item(s) lack answers for the current "
                   "questions; run `flywheel topup --items labeled`.")


@cli.command()
@score_option
@click.pass_context
def status(ctx, score_name):
    """Where the flywheel stands, and what it thinks is worth doing next."""
    workspace = _workspace(ctx)
    score_name = _score_name(workspace, score_name)
    state = loop_status(workspace, score_name)
    console = RichConsole()
    console.print(f"[bold]{score_name}[/bold] · scorecard v{state.version} · "
                  f"{state.n_labeled} labeled ({state.n_effective:.1f} effective)")
    ladder = f"tier [bold]{state.tier}[/bold]"
    if state.next_tier:
        ladder += f" · {state.distance_to_next:.0f} more effective labels to reach {state.next_tier}"
    console.print(ladder)
    for trigger in state.triggers.values():
        table = Table(title=f"{trigger.name}: {'WORTH DOING NOW' if trigger.fire else 'not yet'}",
                      title_justify="left", box=None)
        table.add_column("", width=2)
        table.add_column("condition", no_wrap=True)   # never wrap the name; let "have" wrap
        table.add_column("have")
        table.add_column("need")
        for condition in trigger.conditions:
            table.add_row("[green]✓[/green]" if condition.met else "[red]✗[/red]",
                          condition.name, condition.have, condition.need)
        console.print(table)


@cli.command()
@score_option
@click.option("--version", type=int, default=None, help="Score this scorecard version instead.")
@click.pass_context
def evaluate(ctx, score_name, version):
    """Accuracy on the held-out test split, and alignment with the human."""
    workspace = _workspace(ctx)
    score_name = _score_name(workspace, score_name)
    console = RichConsole()
    board = scoreboard(workspace, score_name, version=version)
    summary = board.summary
    console.print(f"[bold]Scoreboard[/bold] · scorecard v{board.version} · {summary.n} held-out "
                  f"items no human has seen")
    table = Table(box=None)
    for name in ("accuracy", "ECE", "Brier", "mean confidence", "overconfidence"):
        table.add_column(name, justify="right")
    table.add_row(f"{summary.accuracy:.3f}", f"{summary.ece:.3f}", f"{summary.brier:.3f}",
                  f"{summary.mean_confidence:.3f}", f"{summary.overconfidence:+.3f}")
    console.print(table)
    console.print("By tier: " + "  ".join(f"{t} {a:.3f}" for t, a in board.by_tier.items()))

    curve = alignment_curve(workspace, score_name)
    if curve:
        recent = alignment_curve(workspace, score_name, window=20)[-1]
        console.print(
            f"\n[bold]Alignment[/bold] with you over {curve[-1].k} labels: "
            f"{curve[-1].agreement:.1%} overall, {recent.agreement:.1%} over the last "
            f"{min(20, curve[-1].k)} (inverse-propensity weighted; each prediction was made "
            "before you labeled it)")
    else:
        console.print("\nNo labels yet, so no alignment to report. Run `flywheel label`.")


@cli.command()
@score_option
@click.pass_context
def history(ctx, score_name):
    """The scorecard's version lineage, each version scored on the held-out split."""
    workspace = _workspace(ctx)
    score_name = _score_name(workspace, score_name)
    table = Table(title="Scorecard lineage", title_justify="left")
    for name in ("version", "how", "after N labels", "accuracy", "ECE", "Brier"):
        table.add_column(name, justify="right" if name != "how" else "left")
    for point in version_history(workspace, score_name):
        summary = point.scoreboard.summary
        table.add_row(f"v{point.version}", point.kind, str(point.n_feedback),
                      f"{summary.accuracy:.3f}", f"{summary.ece:.3f}", f"{summary.brier:.3f}")
    RichConsole().print(table)


@cli.command()
@click.option("--items", "which", type=click.Choice(["labeled", "pool", "all"]), default="labeled",
              show_default=True, help="Which items to fill in answers for.")
@click.option("--yes", is_flag=True, help="Spend the requests. Without it, only the price is shown.")
@click.option("--concurrency", type=int, default=16, show_default=True)
@click.pass_context
def topup(ctx, which, yes, concurrency):
    """Ask Jev for answers the current scorecard is missing. Prices it first.

    A new element costs one request per item that lacks it, carrying only the missing
    questions, and ten new elements cost the same number of requests as one.
    """
    workspace = _workspace(ctx)
    card = workspace.scorecard()
    questions = card.questions()
    labeled = set().union(*(workspace.labeled_ids(s.name) for s in card.scores))
    pool = {"labeled": [i for i in workspace.items if i.id in labeled],
            "pool": workspace.split("pool"), "all": workspace.items}[which]
    plan = workspace.cache.plan([i.id for i in pool], questions)
    click.echo(f"{len(pool)} items considered; {plan.requests} need a Jev request "
               f"({plan.missing_answers} missing answers).")
    if plan.is_free:
        click.echo("Nothing to fetch.")
        return
    if not yes:
        click.echo("Not spending anything. Re-run with --yes to send them.")
        return

    from dotenv import load_dotenv
    load_dotenv()
    factory = (ctx.obj or {}).get("client_factory")
    session = JevSession(client_factory=factory) if factory else JevSession()
    with click.progressbar(length=plan.requests, label="Asking Jev") as bar:
        async def tick(done, total):
            bar.update(done - bar.pos)

        report = asyncio.run(workspace.cache.fill(
            session, pool, questions, concurrency=concurrency, on_progress=tick))
    click.echo(f"{report.requested} requests sent, {report.failures} failed, "
               f"{report.input_tokens:,} input tokens.")
    if report.failures:
        click.echo("Run it again to retry the failures; completed answers are kept.")


def main():  # pragma: no cover - console-script entry point
    cli(obj={})


if __name__ == "__main__":  # pragma: no cover
    main()
