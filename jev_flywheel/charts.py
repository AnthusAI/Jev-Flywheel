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
from typing import List

from jev_flywheel.report import alignment_curve, complete_items, history, scoreboard
from jev_flywheel.workspace import Workspace

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # categorical slot 1: the latest scorecard
ORANGE = "#eb6834"    # categorical slot 2: Jev's own answer, the baseline


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(GRID)


def _label_versions(ax, xs: List[float], ys: List[float], names: List[str], above: bool = True):
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10 if above else -16),
                    ha="center", fontsize=8, color=INK,
                    # A surface-colored plate keeps the label legible where a line runs under it.
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})


def flywheel_figure(workspace: Workspace, score_name: str, *, split: str = "test"):
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), facecolor=SURFACE)
    fig.suptitle(
        f"Jev plus a head that learns from feedback: {len(items):,} held-out items no human saw",
        x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")

    accuracy = [p.scoreboard.summary.accuracy for p in points]
    ax = axes[0][0]
    _style(ax, "Accuracy", "labels collected", "held-out accuracy")
    ax.step(xs, accuracy, where="post", color=BLUE, linewidth=2)
    ax.plot(xs, accuracy, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2)
    _label_versions(ax, xs, accuracy, names)
    low, high = min(accuracy), max(accuracy)
    pad = max((high - low) * 0.6, 0.01)
    ax.set_ylim(low - pad, high + pad * 1.6)

    ece = [p.scoreboard.summary.ece for p in points]
    ax = axes[0][1]
    _style(ax, "Calibration error (lower is better)", "labels collected", "expected calibration error")
    ax.step(xs, ece, where="post", color=BLUE, linewidth=2)
    ax.plot(xs, ece, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    _label_versions(ax, xs, ece, names)
    ax.set_ylim(0, max(ece) * 1.25)

    ax = axes[1][0]
    _style(ax, "Is \"90% sure\" true 90% of the time?", "stated confidence", "how often it was right")
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
            label="perfectly calibrated")
    for point, color, label in ((points[0], ORANGE, f"{names[0]}: Jev alone"),
                                (points[-1], BLUE, f"{names[-1]}: latest")):
        bins = scoreboard(workspace, score_name, version=point.version, split=split,
                          item_ids=items).bins
        ax.plot([b.mean_confidence for b in bins], [b.accuracy for b in bins], "o-",
                color=color, linewidth=2, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)

    ax = axes[1][1]
    _style(ax, "Agreement with you, over time", "labels collected", "share of predictions you agreed with")
    curve = alignment_curve(workspace, score_name, window=20)
    if curve:
        ax.plot([p.k for p in curve], [p.agreement for p in curve], color=BLUE, linewidth=2,
                label="last 20 labels, weighted")
        for point, name in zip(points[1:], names[1:]):
            ax.axvline(point.n_feedback, color=INK_SECONDARY, linewidth=1, linestyle=(0, (2, 3)))
            ax.annotate(name, (point.n_feedback, 1.0), textcoords="offset points",
                        xytext=(4, -12), fontsize=8, color=INK)
        ax.set_ylim(0, 1.02)
    else:
        ax.text(0.5, 0.5, "no labels yet", ha="center", va="center", color=INK_SECONDARY,
                transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_chart(workspace: Workspace, score_name: str, path: Path, *, split: str = "test",
               dpi: int = 160) -> Path:
    """Render the figure to a PNG."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = flywheel_figure(workspace, score_name, split=split)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
