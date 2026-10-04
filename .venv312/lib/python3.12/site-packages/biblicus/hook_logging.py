"""
Structured hook execution logging.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field

from .hooks import HookPoint
from .time import utc_now_iso


def new_operation_id() -> str:
    """
    Create a new operation identifier for hook log grouping.

    :return: Operation identifier.
    :rtype: str
    """
    return str(uuid.uuid4())


def redact_source_uri(source_uri: str) -> str:
    """
    Redact sensitive components from a source uniform resource identifier.

    :param source_uri: Source uniform resource identifier.
    :type source_uri: str
    :return: Redacted source uniform resource identifier.
    :rtype: str
    """
    parsed = urlparse(source_uri)

    if not parsed.scheme:
        return source_uri

    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]

    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


class HookLogEntry(BaseModel):
    """
    Single structured log record for hook execution.

    :ivar operation_id: Identifier for the enclosing command or call.
    :vartype operation_id: str
    :ivar hook_point: Hook point that executed.
    :vartype hook_point: HookPoint
    :ivar hook_id: Hook implementation identifier.
    :vartype hook_id: str
    :ivar recorded_at: International Organization for Standardization 8601 timestamp for log record creation.
    :vartype recorded_at: str
    :ivar status: Execution status string.
    :vartype status: str
    :ivar message: Optional message describing execution results.
    :vartype message: str or None
    :ivar item_id: Optional item identifier.
    :vartype item_id: str or None
    :ivar relpath: Optional relative path associated with an item.
    :vartype relpath: str or None
    :ivar source_uri: Optional redacted source uniform resource identifier.
    :vartype source_uri: str or None
    :ivar details: Optional structured details about changes.
    :vartype details: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    hook_point: HookPoint
    hook_id: str
    recorded_at: str
    status: str = Field(min_length=1)
    message: Optional[str] = None
    item_id: Optional[str] = None
    relpath: Optional[str] = None
    source_uri: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HookLogger:
    """
    Hook logger that writes JSON lines records to a corpus log directory.

    :ivar log_dir: Directory where log files are written.
    :vartype log_dir: Path
    :ivar operation_id: Operation identifier for grouping records.
    :vartype operation_id: str
    """

    def __init__(self, *, log_dir: Path, operation_id: str):
        """
        Initialize a hook logger.

        :param log_dir: Log directory to write into.
        :type log_dir: Path
        :param operation_id: Operation identifier for grouping records.
        :type operation_id: str
        """
        self.log_dir = log_dir
        self.operation_id = operation_id

    @property
    def path(self) -> Path:
        """
        Return the log file path for this operation.

        :return: Log file path.
        :rtype: Path
        """
        return self.log_dir / f"{self.operation_id}.jsonl"

    def record(
        self,
        *,
        hook_point: HookPoint,
        hook_id: str,
        status: str,
        message: Optional[str] = None,
        item_id: Optional[str] = None,
        relpath: Optional[str] = None,
        source_uri: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a structured hook log record.

        :param hook_point: Hook point that executed.
        :type hook_point: HookPoint
        :param hook_id: Hook identifier.
        :type hook_id: str
        :param status: Status string such as ok, denied, or error.
        :type status: str
        :param message: Optional message describing results.
        :type message: str or None
        :param item_id: Optional item identifier.
        :type item_id: str or None
        :param relpath: Optional relative path for the item.
        :type relpath: str or None
        :param source_uri: Optional source uniform resource identifier.
        :type source_uri: str or None
        :param details: Optional structured details.
        :type details: dict[str, Any] or None
        :return: None.
        :rtype: None
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        entry = HookLogEntry(
            operation_id=self.operation_id,
            hook_point=hook_point,
            hook_id=hook_id,
            recorded_at=utc_now_iso(),
            status=status,
            message=message,
            item_id=item_id,
            relpath=relpath,
            source_uri=redact_source_uri(source_uri) if source_uri else None,
            details=dict(details or {}),
        )
        line = json.dumps(entry.model_dump(), sort_keys=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
