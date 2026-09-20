"""The flywheel's state on disk.

Everything is an append-only file in one directory. Nothing is ever rewritten in
place, so a run is auditable and replayable: every label, every scorecard version and
every fit and rethink stays on the record, and the chart in the README is produced
from that record rather than from anything kept in memory.

    items.jsonl          the corpus (immutable input)
    answers.jsonl        cached Jev answers, keyed per question
    feedback.jsonl       every human judgement, in the order it was given
    events.jsonl         every fit and rethink attempt, with its outcome
    scorecards/
      v1.yaml, v2.yaml   each version of the whole-scorecard YAML
      lineage.jsonl      how each version came to be (seeded, fitted, steered)

A scorecard version is never edited. A change makes a new version with a parent, which
is what makes the flywheel's history a lineage you can read, diff and roll back.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from jev_flywheel.answers import AnswerCache, import_answers_jsonl
from jev_flywheel.items import (
    FeedbackItem, Item, JsonlStore, load_items, normalize_label, now)
from jev_flywheel.scorecard import Scorecard
from jev_flywheel.scoring import ScoreResult, predict
from jev_flywheel.steering import SteeringState


class WorkspaceError(RuntimeError):
    """The workspace is missing, or in a state the requested operation cannot use."""


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._items: Optional[List[Item]] = None
        self._cache: Optional[AnswerCache] = None

    # ---- paths ----------------------------------------------------------------

    @property
    def items_path(self) -> Path:
        return self.root / "items.jsonl"

    @property
    def answers_path(self) -> Path:
        return self.root / "answers.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self.root / "feedback.jsonl"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def scorecards_dir(self) -> Path:
        return self.root / "scorecards"

    @property
    def exists(self) -> bool:
        return self.items_path.exists() and self.scorecards_dir.exists()

    def require(self) -> "Workspace":
        if not self.exists:
            raise WorkspaceError(
                f"no workspace at {self.root}. Run `flywheel init` first, "
                "or point --workspace at one.")
        return self

    # ---- creation -------------------------------------------------------------

    @classmethod
    def init(cls, root: Path, fixtures: Path, *, force: bool = False) -> "Workspace":
        """Create a workspace from the committed fixtures.

        Imports the cached Jev answers under the reference scorecard's question
        bodies, which are the ones they were collected with, and seeds version 1 of
        the scorecard from the fixtures. No network is involved.
        """
        workspace = cls(root)
        if workspace.exists and not force:
            raise WorkspaceError(
                f"{root} already holds a workspace; pass --force to replace it. "
                "That deletes its labels.")
        if workspace.root.exists() and force:
            shutil.rmtree(workspace.root)
        workspace.root.mkdir(parents=True)

        shutil.copyfile(Path(fixtures) / "items.jsonl", workspace.items_path)
        reference = Scorecard.from_yaml(
            (Path(fixtures) / "scorecards" / "reference_full.yaml").read_text())
        import_answers_jsonl(
            Path(fixtures) / "answers.jsonl.gz", reference.questions(),
            AnswerCache(workspace.answers_path))
        seed = Scorecard.from_yaml((Path(fixtures) / "scorecards" / "v1.yaml").read_text())
        workspace.commit_scorecard(seed, kind="seed", provenance={"source": "fixtures/v1.yaml"})
        return workspace

    # ---- items and answers ----------------------------------------------------

    @property
    def items(self) -> List[Item]:
        if self._items is None:
            self._items = load_items(self.items_path)
        return self._items

    def item(self, item_id: str) -> Item:
        for candidate in self.items:
            if candidate.id == item_id:
                return candidate
        raise KeyError(item_id)

    def split(self, name: str) -> List[Item]:
        return [i for i in self.items if i.split == name]

    @property
    def cache(self) -> AnswerCache:
        if self._cache is None:
            self._cache = AnswerCache(self.answers_path)
        return self._cache

    # ---- feedback -------------------------------------------------------------

    @property
    def _feedback_store(self) -> JsonlStore:
        return JsonlStore(self.feedback_path, FeedbackItem)

    def feedback(self) -> List[FeedbackItem]:
        return self._feedback_store.all()

    def add_feedback(self, record: FeedbackItem) -> None:
        self._feedback_store.append(record)

    def labeled_ids(self, score_name: str) -> set:
        """Items that already have a human judgement, so selection never repeats them."""
        return {f.item_id for f in self.feedback() if f.score_name == score_name}

    def n_labeled(self, score_name: str) -> int:
        """How many items carry a usable label. The one place this is counted, so
        the events that record it and the steering state that reads it cannot drift."""
        latest: Dict[str, FeedbackItem] = {}
        for record in self.feedback():
            if record.score_name == score_name:
                latest[record.item_id] = record
        return sum(1 for f in latest.values() if f.label is not None)

    # ---- scorecard lineage ----------------------------------------------------

    def _lineage(self) -> List[Dict[str, Any]]:
        path = self.scorecards_dir / "lineage.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @property
    def version(self) -> int:
        lineage = self._lineage()
        return lineage[-1]["version"] if lineage else 0

    def scorecard(self, version: Optional[int] = None) -> Scorecard:
        version = version or self.version
        path = self.scorecards_dir / f"v{version}.yaml"
        if not path.exists():
            raise WorkspaceError(f"no scorecard version {version} in {self.scorecards_dir}")
        return Scorecard.from_yaml(path.read_text())

    def commit_scorecard(self, card: Scorecard, *, kind: str,
                         provenance: Optional[Dict[str, Any]] = None) -> int:
        """Write a new version. It never overwrites one, and it records its parent."""
        parent = self.version or None
        version = (parent or 0) + 1
        card.version = version
        card.validate()
        text = card.to_yaml()
        self.scorecards_dir.mkdir(parents=True, exist_ok=True)
        (self.scorecards_dir / f"v{version}.yaml").write_text(text)
        entry = {
            "version": version, "parent": parent, "kind": kind, "created_at": now(),
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            # How much feedback existed when this version was made, so a chart of
            # quality against labels can place each version on its x axis.
            "n_feedback": len(self.feedback()),
            "provenance": provenance or {},
        }
        with (self.scorecards_dir / "lineage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return version

    def lineage(self) -> List[Dict[str, Any]]:
        return self._lineage()

    # ---- events ---------------------------------------------------------------

    def log_event(self, kind: str, score_name: str, **data: Any) -> Dict[str, Any]:
        """Record a fit or rethink attempt, with the counts steering needs later."""
        event = {
            "kind": kind, "score_name": score_name, "at": now(), "version": self.version,
            "n_feedback": len(self.feedback()), "n_labeled": self.n_labeled(score_name),
            **data,
        }
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event

    def events(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows = [json.loads(line) for line in self.events_path.read_text().splitlines()
                if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    # ---- prediction and steering state ----------------------------------------

    def predict(self, item_id: str, score_name: str,
                card: Optional[Scorecard] = None) -> ScoreResult:
        """The current scorecard's answer for one item, from cached answers."""
        card = card or self.scorecard()
        answers = self.cache.partial_answers_for(item_id, card.questions())
        return predict(card.score(score_name), answers)

    def steering_state(self, score_name: str, n_effective: float = 0.0) -> SteeringState:
        """What the steering policy needs, derived from the recorded history."""
        feedback = self.feedback()
        n_labeled = self.n_labeled(score_name)
        fits = [e for e in self.events("fit") if e.get("score_name") == score_name]
        rethinks = [e for e in self.events("rethink") if e.get("score_name") == score_name]
        last_fit_labels = fits[-1]["n_labeled"] if fits else 0
        last_rethink = rethinks[-1] if rethinks else None

        def explained_mismatch(record: FeedbackItem) -> bool:
            return (bool(record.edit_comment_value)
                    and normalize_label(record.initial_answer_value)
                    != normalize_label(record.final_answer_value))

        since = feedback[last_rethink["n_feedback"]:] if last_rethink else feedback
        commented = sum(1 for f in since if f.score_name == score_name and f.label is not None
                        and explained_mismatch(f))
        fitted = [e for e in fits if e.get("fitted")]
        return SteeringState(
            n_labeled=n_labeled,
            n_effective=n_effective or float(n_labeled),
            labels_since_fit=n_labeled - last_fit_labels,
            labels_since_rethink=n_labeled - (last_rethink["n_labeled"] if last_rethink else 0),
            commented_mismatches_since_rethink=commented,
            fit_log_losses=[e["log_loss"] for e in fitted if e.get("log_loss") is not None],
            oof_accuracy=fitted[-1].get("oof_accuracy") if fitted else None,
        )
