"""
Virtuus-backed persistence for Reinforcement Memory.

Manages three tables:
- ``texts``: raw ingested :class:`~._models.TimestampedText` records,
  queryable by group and timestamp.
- ``topics``: persisted topic state (label, weight, tier, lifecycle, etc.),
  queryable by group.
- ``runs``: audit trail of analysis runs, queryable by group.

Requires the ``reinforcement-memory`` optional dependency group (``virtuus``).
"""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._models import TimestampedText

_SCHEMA: Dict[str, Any] = {
    "tables": {
        "texts": {
            "primary_key": "id",
            "directory": "texts",
            "gsis": {
                "by_group_timestamp": {
                    "partition_key": "group_id",
                    "sort_key": "timestamp",
                },
            },
        },
        "topics": {
            "primary_key": "topic_id",
            "directory": "topics",
            "gsis": {
                "by_group": {
                    "partition_key": "group_id",
                    "sort_key": "last_updated",
                },
            },
        },
        "runs": {
            "primary_key": "run_id",
            "directory": "runs",
            "gsis": {
                "by_group": {
                    "partition_key": "group_id",
                    "sort_key": "timestamp",
                },
            },
        },
    }
}


class ReinforcementMemoryStore:
    """
    Typed accessor for the Reinforcement Memory Virtuus database.

    Creates and manages the ``texts``, ``topics``, and ``runs`` tables under
    ``data_dir``.  Directories are created automatically on first access.

    :param data_dir: Root directory for all Virtuus JSON files.
    :type data_dir: str
    """

    def __init__(self, data_dir: str) -> None:
        """Initialise the store and load existing data from disk."""
        _ensure_virtuus_version_file()
        try:
            from virtuus import Database
        except ImportError as exc:
            raise ImportError(
                "Reinforcement Memory requires Virtuus. "
                'Install it with: pip install "biblicus[reinforcement-memory]"'
            ) from exc

        os.makedirs(data_dir, exist_ok=True)
        self._db = Database.from_schema_dict(_SCHEMA, data_root=data_dir)

    # ------------------------------------------------------------------
    # Texts
    # ------------------------------------------------------------------

    def put_text(self, text: TimestampedText) -> None:
        """
        Persist a :class:`~._models.TimestampedText` record.

        Idempotent: calling again with the same ``id`` updates the record.

        :param text: Text record to store.
        """
        self._db.tables["texts"].put(
            {
                "id": text.id,
                "group_id": text.group_id,
                "timestamp": text.timestamp,
                "text": text.text,
                "metadata": text.metadata,
            }
        )

    def put_texts(self, texts: List[TimestampedText]) -> int:
        """
        Persist multiple text records.  Returns the count stored.

        :param texts: Iterable of text records to store.
        :return: Number of records stored.
        """
        for t in texts:
            self.put_text(t)
        return len(texts)

    def get_texts_for_group(
        self,
        group_id: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all text records for a group, ordered by timestamp.

        :param group_id: Group identifier.
        :param since: Optional ISO timestamp lower bound (inclusive).
        :param until: Optional ISO timestamp upper bound (inclusive).
        :return: List of raw record dicts.
        """
        table = self._db.tables["texts"]
        sort_condition = _build_range_condition(since, until)
        return table.query_gsi("by_group_timestamp", group_id, sort_condition)

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def put_topic(self, topic: Dict[str, Any]) -> None:
        """
        Persist topic state.

        The ``topic_id`` field is required and used as the primary key.

        :param topic: Topic dict (must contain ``topic_id`` and ``group_id``).
        """
        self._db.tables["topics"].put(topic)

    def get_topics_for_group(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all topic records for a group.

        :param group_id: Group identifier.
        :return: List of topic dicts.
        """
        return self._db.tables["topics"].query_gsi("by_group", group_id)

    def delete_topics_for_group(self, group_id: str) -> int:
        """
        Delete all topic records for a group.

        :param group_id: Group identifier.
        :return: Number of records deleted.
        """
        table = self._db.tables["topics"]
        records = table.query_gsi("by_group", group_id)
        for rec in records:
            table.delete(rec["topic_id"])
        return len(records)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def record_run(self, run: Dict[str, Any]) -> None:
        """
        Append an analysis run record.

        :param run: Run dict (must contain ``run_id`` and ``group_id``).
        """
        self._db.tables["runs"].put(run)

    def get_runs_for_group(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all run records for a group.

        :param group_id: Group identifier.
        :return: List of run dicts ordered by timestamp.
        """
        return self._db.tables["runs"].query_gsi("by_group", group_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_range_condition(since: Optional[str], until: Optional[str]):
    """Build a Virtuus sort condition from optional ISO timestamp bounds."""
    if since is None and until is None:
        return None
    _ensure_virtuus_version_file()
    try:
        from virtuus import Sort
    except ImportError:
        return None

    if since is not None and until is not None:
        return Sort.between(since, until)
    if since is not None:
        return Sort.gte(since)
    return Sort.lte(until)


def _ensure_virtuus_version_file() -> None:
    """
    Ensure the version file expected by current Virtuus wheels exists.

    Virtuus 0.5.0 imports by reading ``<sys.prefix>/lib/VERSION``. Some wheel
    builds omit that file, which causes import-time failure even when the
    package is installed. This function creates the file from installed package
    metadata when missing.
    """
    try:
        installed_version = metadata.version("virtuus")
    except metadata.PackageNotFoundError:
        return
    version_file = Path(sys.prefix) / "lib" / "VERSION"
    if version_file.exists():
        return
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(f"{installed_version}\n", encoding="utf-8")
