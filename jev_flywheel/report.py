"""Measuring the flywheel honestly.

Two numbers, kept apart on purpose:

* **The scoreboard** is accuracy against the corpus's own reference labels on the
  held-out *test* split. No human ever sees those items and no selection rule ever
  touches them, so this is the number that means what it says. It is what the README's
  headline chart plots.
* **Alignment** is agreement with the human on the items the human was shown. Those
  were chosen because they teach the most, so they are a biased sample of the corpus.
  It is measured *prequentially*: the prediction shown to the human was made before that
  label existed, so each agree/disagree is an honest out-of-sample trial with no
  train/test split needed. Inverse-propensity weights make it estimate agreement over
  the whole pool instead of over the hard items selection preferred.

They can move independently, and that is a finding rather than a bug. The corpus's weak
and neutral tiers have arbitrary labels. If steering toward a human's intent lifts
alignment while test accuracy stays flat, the human was *defining* the label, not
confirming it, and the article should say so.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from jev_flywheel.evaluate import Bin, Summary, reliability_bins, summarize
from jev_flywheel.fit import latest_feedback
from jev_flywheel.items import agrees
from jev_flywheel.sampling import inverse_propensity_weights, kish_n_effective
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import predict
from jev_flywheel.workspace import Workspace


@dataclass
class Scoreboard:
    """A scorecard's accuracy and calibration on a held-out split."""

    version: int
    split: str
    summary: Summary
    by_tier: Dict[str, float] = field(default_factory=dict)
    bins: List[Bin] = field(default_factory=list)
    # Share of the split whose answers were all on file. Below 1 the scorecard was served
    # with missing features, so its numbers understate what it would do given answers.
    coverage: float = 1.0

    @property
    def accuracy(self) -> float:
        return self.summary.accuracy


def scoreboard(workspace: Workspace, score_name: str, card: Optional[Scorecard] = None,
               *, split: str = "test", version: Optional[int] = None,
               item_ids: Optional[Set[str]] = None) -> Scoreboard:
    """Accuracy and calibration against the corpus's reference labels.

    ``item_ids`` restricts scoring to a fixed set of items, so that several scorecard
    versions can be compared on exactly the same held-out items. Without it, a version
    that asks a new question is scored on items that may lack its answers.
    """
    version = version or workspace.version
    card = card or workspace.scorecard(version)
    score = card.score(score_name)
    items = [i for i in workspace.split(split) if i.reference_label is not None
             and (item_ids is None or i.id in item_ids)]
    answers = workspace.cache.bulk_partial_answers([i.id for i in items], card.questions())

    confidences: List[float] = []
    correct: List[int] = []
    tiers: Dict[str, List[int]] = {}
    complete = 0
    wanted = set(card.questions())
    for item in items:
        complete += wanted <= set(answers[item.id])
        result = predict(score, answers[item.id])
        hit = int(agrees(result.value, item.reference_label))
        confidences.append(result.confidence or 0.0)
        correct.append(hit)
        tiers.setdefault(str(item.metadata.get("tier", "?")), []).append(hit)
    return Scoreboard(
        version=version, split=split, summary=summarize(confidences, correct),
        by_tier={t: sum(v) / len(v) for t, v in sorted(tiers.items())},
        bins=reliability_bins(confidences, correct),
        coverage=complete / len(items) if items else 1.0)


@dataclass
class AlignmentPoint:
    """Running weighted agreement after the k-th label."""

    k: int
    agreement: float
    n_effective: float


def alignment_curve(workspace: Workspace, score_name: str,
                    window: int = 0) -> List[AlignmentPoint]:
    """Prequential, inverse-propensity-weighted agreement with the human over time.

    ``window`` of 0 uses everything so far; a positive window uses only the most recent
    labels, which shows the trend rather than the long-run average.
    """
    records = [f for f in latest_feedback(workspace.feedback(), score_name).values()
               if f.label is not None and f.is_agreement is not None and f.propensity]
    records.sort(key=lambda f: f.edited_at or "")
    curve: List[AlignmentPoint] = []
    for k in range(1, len(records) + 1):
        seen = records[max(0, k - window):k] if window else records[:k]
        weights = inverse_propensity_weights([f.propensity for f in seen], max_ratio=20.0)
        agreement = sum(w * int(bool(f.is_agreement)) for w, f in zip(weights, seen)) / sum(weights)
        curve.append(AlignmentPoint(k, agreement, kish_n_effective(weights)))
    return curve


@dataclass
class VersionPoint:
    """One scorecard version and how it did on the scoreboard."""

    version: int
    kind: str
    n_feedback: int
    scoreboard: Scoreboard


def complete_items(workspace: Workspace, split: str = "test",
                   version: Optional[int] = None) -> Set[str]:
    """Items in a split that have every answer a scorecard version asks for."""
    questions = workspace.scorecard(version).questions()
    return {i.id for i in workspace.split(split)
            if all(workspace.cache.get(i.id, n, q) is not None for n, q in questions.items())}


def history(workspace: Workspace, score_name: str, split: str = "test",
            item_ids: Optional[Set[str]] = None) -> List[VersionPoint]:
    """The scoreboard for every scorecard version, placed by how much feedback existed.

    Pass ``item_ids`` (see ``complete_items``) to compare every version on the same items.
    """
    return [
        VersionPoint(entry["version"], entry["kind"], entry.get("n_feedback", 0),
                     scoreboard(workspace, score_name, version=entry["version"], split=split,
                                item_ids=item_ids))
        for entry in workspace.lineage()
    ]


def confusion(workspace: Workspace, score_name: str,
              card: Optional[Scorecard] = None) -> Tuple[List[str], List[List[int]]]:
    """Predicted against reference label on the test split, as ``(labels, matrix)``."""
    card = card or workspace.scorecard()
    score = card.score(score_name)
    items = [i for i in workspace.split("test") if i.reference_label]
    answers = workspace.cache.bulk_partial_answers([i.id for i in items], card.questions())
    pairs = [(i.reference_label, predict(score, answers[i.id]).value) for i in items]
    labels = sorted({a for a, _ in pairs} | {p for _, p in pairs})
    matrix = [[sum(1 for a, p in pairs if a == row and p == col) for col in labels]
              for row in labels]
    return labels, matrix
