"""Items, feedback, and the label normalization rules.

The vocabulary here deliberately matches Plexus so that moving a working
experiment into Plexus is a port and not a rewrite. Plexus's own Python models
mirror its GraphQL schema and so use camelCase; this project uses snake_case
because it has no GraphQL layer. The mapping is one-to-one:

    item_id              <- FeedbackItem.itemId
    initial_answer_value <- FeedbackItem.initialAnswerValue
    final_answer_value   <- FeedbackItem.finalAnswerValue
    edit_comment_value   <- FeedbackItem.editCommentValue
    editor_name          <- FeedbackItem.editorName
    edited_at            <- FeedbackItem.editedAt
    is_agreement         <- FeedbackItem.isAgreement
    cache_key            <- FeedbackItem.cacheKey

Storage is append-only JSONL. There is no database, no API and no account
scoping; those are the things you graduate to Plexus for.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

# Label provenance, spelled exactly as Plexus spells it, so a ported dataset
# keeps its meaning. Only the first two are safe to train on: the third falls
# back to the AI's own previous prediction, and training on that teaches the
# model to imitate the incumbent champion.
LABEL_SOURCE_VETTED = "vetted_feedback"
LABEL_SOURCE_FINAL = "regular_final_feedback"
LABEL_SOURCE_SCORE_RESULT_OR_IMPORTED = "score_result_or_imported"
LABEL_SOURCE_UNRESOLVED = "unresolved"

TRUSTED_LABEL_SOURCES = (LABEL_SOURCE_VETTED, LABEL_SOURCE_FINAL)

_EMPTY_LABELS = ("", "nan", "n/a", "none", "null")


def now() -> str:
    """An ISO-8601 UTC timestamp. One definition, so records sort lexically."""
    return datetime.now(timezone.utc).isoformat()


def normalize_prediction(value: Any) -> str:
    """Normalize a predicted value the way Plexus's Evaluation does.

    Mirrors Evaluation.py: lowercase, strip, collapse internal whitespace.
    """
    return " ".join(str(value).lower().strip().split())


def normalize_label(value: Any) -> str:
    """Normalize a human label the way Plexus's Evaluation does.

    Mirrors Evaluation.py: lowercase, strip trailing '.!?', then map the
    several spellings of "missing" onto one. Getting this wrong is the classic
    way to send every metric silently to zero.
    """
    if value is None:
        return ""
    text = str(value).lower().strip().rstrip(".!?").strip()
    if text == "nan":
        return ""
    if text == "n/a":
        return "na"
    return text


def agrees(prediction: Any, label: Any) -> bool:
    """Whether a prediction counts as correct against a label.

    Plexus compares after a second normalization pass in which every spelling
    of "missing" collapses to 'na', so '' and 'nan' and None all agree with
    each other. Exact string equality after that.
    """
    predicted = normalize_prediction(prediction)
    actual = normalize_label(label)
    predicted = "na" if predicted in _EMPTY_LABELS else predicted
    actual = "na" if actual in _EMPTY_LABELS else actual
    return predicted == actual


@dataclass
class Item:
    """One thing to be scored.

    `identifiers` is a list of {name, value, url} objects, which is the shape
    Plexus uses everywhere for human-facing IDs.
    """

    id: str
    text: str
    external_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    identifiers: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def split(self) -> Optional[str]:
        """Which corpus split this item belongs to, if the corpus has splits.

        The split lives in metadata rather than as a column because it is a
        property of the dataset, not of the item.
        """
        value = self.metadata.get("split")
        return str(value) if value is not None else None

    @property
    def reference_label(self) -> Optional[str]:
        """The corpus's own ground-truth label, when there is one.

        This is the independent scoreboard. It is never the steering signal --
        that comes from FeedbackItems -- and it must never be read for an item
        the labeling console selected.
        """
        value = self.metadata.get("reference_label")
        return str(value) if value is not None else None


@dataclass
class FeedbackItem:
    """One human judgement about one prediction on one item.

    `initial_answer_value` is what we predicted; `final_answer_value` is what
    the human settled on. They are equal when the human agreed. The comment is
    the part that matters most for this project: it is what the meta-cognition
    step reads to work out which factor we are missing.
    """

    id: str
    item_id: str
    score_name: str
    initial_answer_value: Optional[str] = None
    final_answer_value: Optional[str] = None
    edit_comment_value: Optional[str] = None
    editor_name: Optional[str] = None
    edited_at: Optional[str] = None
    is_agreement: Optional[bool] = None
    cache_key: Optional[str] = None
    label_source: str = LABEL_SOURCE_FINAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> Optional[str]:
        """The label to train on, or None if this item carries no usable one."""
        if self.is_invalid:
            return None
        if self.label_source not in TRUSTED_LABEL_SOURCES:
            return None
        return self.final_answer_value

    @property
    def is_invalid(self) -> bool:
        return bool(self.metadata.get("is_invalid"))

    @property
    def propensity(self) -> Optional[float]:
        """The probability the selection rule gave this item when it was shown.

        Active selection makes the labeled set a biased sample, so fitting has
        to inverse-probability-weight by this. A label with no recorded
        propensity cannot be weighted, which is why the console always writes
        one.
        """
        value = self.metadata.get("propensity")
        return float(value) if value is not None else None

    @property
    def confusion_cell(self) -> Optional[str]:
        """The (predicted, actual) cell this feedback lands in.

        Plexus samples feedback per cell, which is what makes the training
        class balance synthetic and the IPW correction necessary.
        """
        if self.initial_answer_value is None or self.final_answer_value is None:
            return None
        return f"{normalize_label(self.initial_answer_value)}->{normalize_label(self.final_answer_value)}"


def _from_row(cls, row: Dict[str, Any]):
    """Build a dataclass from a stored row, ignoring fields it does not declare.

    Tolerating unknown keys means an older fixture file still loads after a
    field is added, which matters because fixtures are committed.
    """
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in row.items() if k in known})


def load_items(path: Path) -> List[Item]:
    """Read the item corpus. Items are immutable input, so this is a plain read."""
    return JsonlStore(path, Item).all()


class JsonlStore:
    """An append-only JSONL file of dataclass records.

    Append-only because the flywheel's history is the interesting part: every
    label, every scorecard version and every fit stays on the record. Nothing
    is ever rewritten in place, so a run is auditable and replayable.
    """

    def __init__(self, path: Path, record_type: type):
        self.path = Path(path)
        self.record_type = record_type

    def append(self, record) -> None:
        self.append_all([record])

    def append_all(self, records: Iterable[Any]) -> None:
        rows = [json.dumps(asdict(r), ensure_ascii=False) for r in records]
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row + "\n")
            handle.flush()

    def __iter__(self) -> Iterator[Any]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield _from_row(self.record_type, json.loads(line))

    def all(self) -> List[Any]:
        return list(self)

    def latest_by(self, key: str) -> Dict[Any, Any]:
        """The last record for each distinct value of `key`.

        Later records win, which is how an append-only log expresses an update:
        a human revisiting an item writes a second FeedbackItem rather than
        editing the first.
        """
        out: Dict[Any, Any] = {}
        for record in self:
            out[getattr(record, key)] = record
        return out
