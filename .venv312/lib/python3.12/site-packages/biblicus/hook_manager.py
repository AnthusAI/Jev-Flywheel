"""
Hook manager for executing configured lifecycle hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from .constants import CORPUS_DIR_NAME, HOOK_LOGS_DIR_NAME
from .hook_logging import HookLogger, new_operation_id
from .hooks import (
    HookContext,
    HookPoint,
    HookSpec,
    IngestHookContext,
    IngestMutation,
    LifecycleHook,
    build_builtin_hook,
)
from .time import utc_now_iso


class HookManager:
    """
    Hook manager that executes configured hooks and records execution.

    :ivar corpus_uri: Canonical uniform resource identifier for the corpus.
    :vartype corpus_uri: str
    :ivar log_dir: Directory where hook logs are recorded.
    :vartype log_dir: object
    :ivar operation_id: Identifier for this hook execution session.
    :vartype operation_id: str
    """

    def __init__(
        self,
        *,
        corpus_uri: str,
        log_dir: Path,
        hooks: Iterable[LifecycleHook],
        operation_id: Optional[str] = None,
    ):
        """
        Initialize a hook manager.

        :param corpus_uri: Canonical uniform resource identifier for the corpus.
        :type corpus_uri: str
        :param log_dir: Directory where hook logs are written.
        :type log_dir: object
        :param hooks: Hook instances to execute.
        :type hooks: Iterable[LifecycleHook]
        :param operation_id: Optional operation identifier override.
        :type operation_id: str or None
        """
        self.corpus_uri = corpus_uri
        self.log_dir = log_dir
        self.operation_id = operation_id or new_operation_id()
        self._hooks = list(hooks)
        self._logger = HookLogger(log_dir=self.log_dir, operation_id=self.operation_id)

    @classmethod
    def from_config(
        cls, *, corpus_root: Path, corpus_uri: str, hook_specs: Iterable[HookSpec]
    ) -> "HookManager":
        """
        Build a hook manager from config data.

        :param corpus_root: Corpus root directory.
        :type corpus_root: Path
        :param corpus_uri: Canonical uniform resource identifier for the corpus.
        :type corpus_uri: str
        :param hook_specs: Hook specifications loaded from config.
        :type hook_specs: Iterable[HookSpec]
        :return: Hook manager.
        :rtype: HookManager
        :raises KeyError: If a hook identifier is unknown.
        """
        log_dir = corpus_root / CORPUS_DIR_NAME / HOOK_LOGS_DIR_NAME
        hooks: List[LifecycleHook] = []

        for spec in hook_specs:
            hooks.append(build_builtin_hook(spec))

        return cls(corpus_uri=corpus_uri, log_dir=log_dir, hooks=hooks)

    def run_ingest_hooks(
        self,
        *,
        hook_point: HookPoint,
        filename: Optional[str],
        media_type: str,
        title: Optional[str],
        tags: List[str],
        metadata: Dict[str, Any],
        source_uri: str,
        item_id: Optional[str] = None,
        relpath: Optional[str] = None,
    ) -> IngestMutation:
        """
        Run ingestion hooks for a hook point.

        :param hook_point: Hook point to execute.
        :type hook_point: HookPoint
        :param filename: Suggested filename.
        :type filename: str or None
        :param media_type: Media type for the item.
        :type media_type: str
        :param title: Optional title.
        :type title: str or None
        :param tags: Tags associated with the item.
        :type tags: list[str]
        :param metadata: Metadata mapping.
        :type metadata: dict[str, Any]
        :param source_uri: Source uniform resource identifier.
        :type source_uri: str
        :param item_id: Optional item identifier.
        :type item_id: str or None
        :param relpath: Optional relative path.
        :type relpath: str or None
        :return: Combined ingestion mutation result.
        :rtype: IngestMutation
        :raises ValueError: If ingestion is denied by a hook.
        """
        context = IngestHookContext(
            hook_point=hook_point,
            operation_id=self.operation_id,
            corpus_uri=self.corpus_uri,
            created_at=utc_now_iso(),
            filename=filename,
            media_type=media_type,
            title=title,
            tags=list(tags),
            metadata=dict(metadata),
            source_uri=source_uri,
            item_id=item_id,
            relpath=relpath,
        )

        combined = IngestMutation()
        for hook in self._hooks_for_point(hook_point):
            result_dict = self._run_single(hook=hook, context=context)
            mutation = IngestMutation.model_validate(result_dict)
            if mutation.deny:
                self._logger.record(
                    hook_point=hook_point,
                    hook_id=hook.hook_id,
                    status="denied",
                    message=mutation.deny_reason or mutation.message,
                    item_id=item_id,
                    relpath=relpath,
                    source_uri=source_uri,
                    details={"add_tags": mutation.add_tags},
                )
                raise ValueError(mutation.deny_reason or "Ingest denied")
            if mutation.add_tags:
                combined.add_tags.extend(mutation.add_tags)
            self._logger.record(
                hook_point=hook_point,
                hook_id=hook.hook_id,
                status="ok",
                message=mutation.message,
                item_id=item_id,
                relpath=relpath,
                source_uri=source_uri,
                details={"add_tags": mutation.add_tags},
            )

        deduplicated_tags: List[str] = []
        for tag in combined.add_tags:
            if tag not in deduplicated_tags:
                deduplicated_tags.append(tag)
        combined.add_tags = deduplicated_tags
        return combined

    def _hooks_for_point(self, hook_point: HookPoint) -> List[LifecycleHook]:
        eligible: List[LifecycleHook] = []
        for hook in self._hooks:
            if hook_point in list(getattr(hook, "hook_points", [])):
                eligible.append(hook)
        return eligible

    def _run_single(self, *, hook: LifecycleHook, context: HookContext) -> Dict[str, Any]:
        """
        Run a single hook with error capture.

        :param hook: Hook to execute.
        :type hook: LifecycleHook
        :param context: Hook context.
        :type context: HookContext
        :return: Hook result mapping.
        :rtype: dict[str, Any]
        :raises ValueError: If a hook raises an exception.
        """
        try:
            result = hook.run(context)
        except Exception as exc:
            raise ValueError(f"Hook {hook.hook_id!r} failed: {exc}") from exc
        if isinstance(result, BaseModel):
            return result.model_dump()
        raise ValueError(f"Hook {hook.hook_id!r} returned a non-Pydantic result")
