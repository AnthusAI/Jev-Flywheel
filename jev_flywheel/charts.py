"""The flywheel figure: what improved, and when.

Four small panels, one measure each (never a dual axis: accuracy and calibration error are
different scales, so they get their own panels):

    accuracy by scorecard version      the held-out scoreboard, placed by how many labels
    calibration error by version       existed when each version was made
    reliability, first vs latest       is "90% sure" true 90% of the time?
    alignment with the human           agreement over the labels, weighted by propensity

Every version is scored on *the same held-out items* (the ones that have every answer the
latest scorecard asks for), so the lines compare like with like. The colors are the first two
slots of a validated categorical palette; text is set in ink rather than in the series
color, and the version points are labeled directly.

matplotlib is an optional dependency: ``pip install 'jev-flywheel[charts]'``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

from jev_flywheel.report import alignment_curve, complete_items, history, scoreboard
from jev_flywheel.workspace import Workspace

class Theme(NamedTuple):
    """Surface, ink and series colours for one colour scheme.

    Surfaces and ink match the d2 diagrams exactly, so the README does not mix a warm black
    chart with cool black diagrams: these are d2's own light canvas and its Dark Mauve
    (Catppuccin Mocha) canvas, read out of the rendered SVGs.

    The series colours are *selected*, not derived: the dark pair is the same two hues
    re-stepped for a dark surface, and both pairs were checked for colour-vision separation and
    for contrast against the exact canvas they are drawn on. Inverting a light palette is what
    produces unreadable dark charts.
    """

    surface: str
    ink: str
    ink_secondary: str
    grid: str
    blue: str       # categorical slot 1: the latest scorecard
    orange: str     # categorical slot 2: Jev's own answer, the baseline


# Canvas and ink are d2's, so the chart and the diagrams sit in the same palette.
LIGHT = Theme(surface="#FFFFFF", ink="#0A0F25", ink_secondary="#676C7E", grid="#DEE1EB",
              blue="#2a78d6", orange="#eb6834")
DARK = Theme(surface="#1E1E2E", ink="#CDD6F4", ink_secondary="#BAC2DE", grid="#45475A",
             blue="#3987e5", orange="#d95926")


def _style(ax, t: Theme, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(t.surface)
    ax.set_title(title, loc="left", fontsize=11, color=t.ink, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=t.ink_secondary)
    ax.set_ylabel(ylabel, fontsize=9, color=t.ink_secondary)
    ax.tick_params(colors=t.ink_secondary, labelsize=8, length=0)
    ax.grid(True, color=t.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(t.grid)


def _label_versions(ax, t: Theme, xs: List[float], ys: List[float], names: List[str],
                    above: bool = True):
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10 if above else -16),
                    ha="center", fontsize=8, color=t.ink,
                    # A surface-coloured plate keeps the label legible where a line runs under it.
                    bbox={"facecolor": t.surface, "edgecolor": "none", "pad": 1.5})


def flywheel_figure(workspace: Workspace, score_name: str, *, split: str = "test",
                    theme: Theme = LIGHT):
    """Build the four-panel figure. Returns a matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = complete_items(workspace, split)
    if not items:
        raise ValueError("no held-out item has every answer the latest scorecard asks for; "
                         "run `flywheel topup --items test --yes` first")
    points = history(workspace, score_name, split=split, item_ids=items)
    xs = [p.n_feedback for p in points]
    names = [f"v{p.version}" for p in points]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), facecolor=theme.surface)
    fig.suptitle(
        f"Jev plus a head that learns from feedback: {len(items):,} held-out items no human saw",
        x=0.06, ha="left", fontsize=12.5, color=theme.ink, fontweight="bold")

    accuracy = [p.scoreboard.summary.accuracy for p in points]
    ax = axes[0][0]
    _style(ax, theme, "Accuracy", "labels collected", "held-out accuracy")
    ax.step(xs, accuracy, where="post", color=theme.blue, linewidth=2)
    ax.plot(xs, accuracy, "o", color=theme.blue, markersize=8, markeredgecolor=theme.surface,
            markeredgewidth=2)
    _label_versions(ax, theme, xs, accuracy, names)
    low, high = min(accuracy), max(accuracy)
    pad = max((high - low) * 0.6, 0.01)
    ax.set_ylim(low - pad, high + pad * 1.6)

    ece = [p.scoreboard.summary.ece for p in points]
    ax = axes[0][1]
    _style(ax, theme, "Calibration error (lower is better)", "labels collected", "expected calibration error")
    ax.step(xs, ece, where="post", color=theme.blue, linewidth=2)
    ax.plot(xs, ece, "o", color=theme.blue, markersize=8, markeredgecolor=theme.surface, markeredgewidth=2)
    _label_versions(ax, theme, xs, ece, names)
    ax.set_ylim(0, max(ece) * 1.25)

    ax = axes[1][0]
    _style(ax, theme, "Is \"90% sure\" true 90% of the time?", "stated confidence", "how often it was right")
    ax.plot([0, 1], [0, 1], color=theme.ink_secondary, linewidth=1, linestyle=(0, (4, 3)),
            label="perfectly calibrated")
    for point, color, label in ((points[0], theme.orange, f"{names[0]}: Jev alone"),
                                (points[-1], theme.blue, f"{names[-1]}: latest")):
        bins = scoreboard(workspace, score_name, version=point.version, split=split,
                          item_ids=items).bins
        ax.plot([b.mean_confidence for b in bins], [b.accuracy for b in bins], "o-",
                color=color, linewidth=2, markersize=7, markeredgecolor=theme.surface,
                markeredgewidth=1.5, label=label)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=theme.ink)

    ax = axes[1][1]
    _style(ax, theme, "Agreement with you, over time", "labels collected", "share of predictions you agreed with")
    curve = alignment_curve(workspace, score_name, window=20)
    if curve:
        ax.plot([p.k for p in curve], [p.agreement for p in curve], color=theme.blue, linewidth=2,
                label="last 20 labels, weighted")
        for point, name in zip(points[1:], names[1:]):
            ax.axvline(point.n_feedback, color=theme.ink_secondary, linewidth=1, linestyle=(0, (2, 3)))
            ax.annotate(name, (point.n_feedback, 1.0), textcoords="offset points",
                        xytext=(4, -12), fontsize=8, color=theme.ink)
        ax.set_ylim(0, 1.02)
    else:
        ax.text(0.5, 0.5, "no labels yet", ha="center", va="center", color=theme.ink_secondary,
                transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def dark_path(path: Path) -> Path:
    """Where the dark companion of ``path`` goes: ``results.png`` -> ``results-dark.png``."""
    path = Path(path)
    return path.with_name(f"{path.stem}-dark{path.suffix}")


def save_chart(workspace: Workspace, score_name: str, path: Path, *, split: str = "test",
               dpi: int = 160, both_schemes: bool = True) -> Path:
    """Render the figure, by default in both colour schemes.

    Writes ``path`` for light and ``path``-dark for dark, so a README can offer both from one
    ``<picture>`` element and let the reader's browser choose. Returns the light path.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = [(path, LIGHT)] + ([(dark_path(path), DARK)] if both_schemes else [])
    for target, theme in targets:
        fig = flywheel_figure(workspace, score_name, split=split, theme=theme)
        fig.savefig(target, dpi=dpi, facecolor=theme.surface)
        plt.close(fig)
    return path
