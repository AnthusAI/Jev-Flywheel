"""
Lifecycle hook interfaces and built-in hook implementations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class HookPoint(str, Enum):
    """
    Canonical lifecycle hook points for corpus operations.

    :cvar before_ingest: Called before an item is ingested.
    :cvar after_ingest: Called after an item is ingested and indexed.
    :cvar before_reindex: Called before a catalog rebuild starts.
    :cvar after_reindex: Called after a catalog rebuild completes.
    :cvar before_build_snapshot: Called before a retriever snapshot build starts.
    :cvar after_build_snapshot: Called after a retriever snapshot build completes.
    :cvar before_query: Called before a query is executed.
    :cvar after_query: Called after a query completes.
    :cvar before_evaluate_run: Called before an evaluation starts.
    :cvar after_evaluate_run: Called after an evaluation completes.
    """

    before_ingest = "before_ingest"
    after_ingest = "after_ingest"
    before_reindex = "before_reindex"
    after_reindex = "after_reindex"
    before_build_snapshot = "before_build_snapshot"
    after_build_snapshot = "after_build_snapshot"
    before_query = "before_query"
    after_query = "after_query"
    before_evaluate_run = "before_evaluate_run"
    after_evaluate_run = "after_evaluate_run"


class HookSpec(BaseModel):
    """
    On-disk hook specification stored in a corpus config.

    :ivar hook_id: Identifier used to locate a hook implementation.
    :vartype hook_id: str
    :ivar hook_points: Hook points where the hook executes.
    :vartype hook_points: list[HookPoint]
    :ivar config: Hook-specific configuration values.
    :vartype config: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    hook_id: str = Field(min_length=1)
    hook_points: List[HookPoint] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class HookContext(BaseModel):
    """
    Base context passed to hooks.

    :ivar hook_point: Hook point currently executing.
    :vartype hook_point: HookPoint
    :ivar operation_id: Identifier for the enclosing command or call.
    :vartype operation_id: str
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus.
    :vartype corpus_uri: str
    :ivar created_at: International Organization for Standardization 8601 timestamp when the context was created.
    :vartype created_at: str
    """

    model_config = ConfigDict(extra="forbid")

    hook_point: HookPoint
    operation_id: str
    corpus_uri: str
    created_at: str


class IngestHookContext(HookContext):
    """
    Hook context for ingestion hooks.

    :ivar filename: Suggested filename for the item.
    :vartype filename: str or None
    :ivar media_type: Media type for the item.
    :vartype media_type: str
    :ivar title: Optional title associated with the item.
    :vartype title: str or None
    :ivar tags: Tags associated with the item.
    :vartype tags: list[str]
    :ivar metadata: Metadata mapping associated with the item.
    :vartype metadata: dict[str, Any]
    :ivar source_uri: Source uniform resource identifier.
    :vartype source_uri: str
    :ivar item_id: Item identifier when available.
    :vartype item_id: str or None
    :ivar relpath: Relative path to stored raw bytes when available.
    :vartype relpath: str or None
    """

    filename: Optional[str] = None
    media_type: str
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_uri: str
    item_id: Optional[str] = None
    relpath: Optional[str] = None


class HookResult(BaseModel):
    """
    Base hook result with optional message fields.

    :ivar message: Optional human-readable message.
    :vartype message: str or None
    """

    model_config = ConfigDict(extra="forbid")

    message: Optional[str] = None


class IngestMutation(HookResult):
    """
    Hook result describing ingestion mutations.

    :ivar deny: Whether ingest should be denied.
    :vartype deny: bool
    :ivar deny_reason: Optional reason for denial.
    :vartype deny_reason: str or None
    :ivar add_tags: Tags to add.
    :vartype add_tags: list[str]
    """

    deny: bool = False
    deny_reason: Optional[str] = None
    add_tags: List[str] = Field(default_factory=list)


class LifecycleHook:
    """
    Base class for a lifecycle hook implementation.

    :param context: Validated hook context.
    :type context: HookContext
    :return: Hook result. Concrete hook points may require a more specific result type.
    :rtype: HookResult
    """

    hook_id: str
    hook_points: Sequence[HookPoint]

    def run(self, context: HookContext) -> HookResult:
        """
        Execute the hook.

        :param context: Hook context.
        :type context: HookContext
        :return: Hook result.
        :rtype: HookResult
        :raises NotImplementedError: If the hook does not implement run.
        """
        _ = context
        raise NotImplementedError("LifecycleHook.run must be implemented by concrete hooks")


class AddTagsHook:
    """
    Built-in hook that adds tags during ingestion.

    :ivar hook_id: Hook identifier.
    :vartype hook_id: str
    :ivar hook_points: Hook points where the hook applies.
    :vartype hook_points: list[HookPoint]
    :ivar tags: Tags to add.
    :vartype tags: list[str]
    """

    hook_id = "add-tags"

    def __init__(self, *, hook_points: Sequence[HookPoint], tags: Sequence[str]):
        """
        Initialize the add-tags hook.

        :param hook_points: Hook points where the hook runs.
        :type hook_points: Sequence[HookPoint]
        :param tags: Tags to add.
        :type tags: Sequence[str]
        """
        self.hook_points = list(hook_points)
        self.tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

    def run(self, context: HookContext) -> HookResult:
        """
        Run the hook.

        :param context: Hook context.
        :type context: HookContext
        :return: Ingest mutation result.
        :rtype: HookResult
        """
        _ = context
        return IngestMutation(add_tags=list(self.tags))


class DenyAllHook:
    """
    Built-in hook that denies every ingest.

    :ivar hook_id: Hook identifier.
    :vartype hook_id: str
    :ivar hook_points: Hook points where the hook applies.
    :vartype hook_points: list[HookPoint]
    """

    hook_id = "deny-all"

    def __init__(self, *, hook_points: Sequence[HookPoint]):
        """
        Initialize the deny-all hook.

        :param hook_points: Hook points where the hook runs.
        :type hook_points: Sequence[HookPoint]
        """
        self.hook_points = list(hook_points)

    def run(self, context: HookContext) -> HookResult:
        """
        Run the hook.

        :param context: Hook context.
        :type context: HookContext
        :return: Ingest denial result.
        :rtype: HookResult
        """
        _ = context
        return IngestMutation(deny=True, deny_reason="Ingest denied by deny-all hook")


def build_builtin_hook(spec: HookSpec) -> LifecycleHook:
    """
    Build a built-in hook from a hook specification.

    :param spec: Hook specification.
    :type spec: HookSpec
    :return: Hook instance.
    :rtype: LifecycleHook
    :raises KeyError: If the hook identifier is unknown.
    """
    if spec.hook_id == AddTagsHook.hook_id:
        tags = spec.config.get("tags") or []
        return AddTagsHook(
            hook_points=spec.hook_points, tags=tags if isinstance(tags, list) else []
        )
    if spec.hook_id == DenyAllHook.hook_id:
        return DenyAllHook(hook_points=spec.hook_points)
    raise KeyError(f"Unknown hook_id {spec.hook_id!r}")
