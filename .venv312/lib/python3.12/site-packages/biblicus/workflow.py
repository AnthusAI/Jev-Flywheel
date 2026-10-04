"""
Task planning utilities for Biblicus workflows.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .corpus import Corpus
from .extraction import build_extraction_snapshot, create_extraction_configuration_manifest
from .models import ExtractionSnapshotListEntry, RetrievalSnapshot
from .retrieval import create_configuration_manifest
from .retrievers import get_retriever

TASK_KIND_ALIASES = {
    "fetch": "load",
    "sync": "load",
    "build": "index",
    "run": "query",
}


class Task(BaseModel):
    """
    Planned task with dependency metadata.

    :ivar name: User-facing task name.
    :vartype name: str
    :ivar kind: Canonical task kind.
    :vartype kind: str
    :ivar target_type: Target type (corpus or retriever).
    :vartype target_type: str
    :ivar target_id: Identifier for the task target.
    :vartype target_id: str
    :ivar depends_on: Dependent tasks that must run first.
    :vartype depends_on: list[Task]
    :ivar status: Current task status (ready, complete, blocked).
    :vartype status: str
    :ivar reason: Optional reason when blocked.
    :vartype reason: str or None
    :ivar metadata: Task-specific configuration metadata.
    :vartype metadata: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    target_type: str
    target_id: str
    depends_on: List["Task"] = Field(default_factory=list)
    status: str = Field(default="ready")
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


Task.model_rebuild()


class Plan(BaseModel):
    """
    Ordered task plan with dependency resolution results.

    :ivar tasks: Ordered list of tasks to execute.
    :vartype tasks: list[Task]
    :ivar root: Root task for the plan.
    :vartype root: Task
    :ivar status: Aggregate plan status (ready, complete, blocked).
    :vartype status: str
    :ivar missing_handlers: Task kinds without a registered handler.
    :vartype missing_handlers: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    tasks: List[Task]
    root: Task
    status: str = "ready"
    missing_handlers: List[str] = Field(default_factory=list)

    def execute(
        self,
        *,
        mode: str = "prompt",
        handler_registry: Optional[Dict[str, Callable[[Task], Any]]] = None,
        prompt_handler: Optional[Callable[[Task], bool]] = None,
    ) -> List[Any]:
        """
        Execute the plan in dependency order.

        :param mode: Execution mode (prompt, auto, dry_run).
        :type mode: str
        :param handler_registry: Mapping of task kinds to callables.
        :type handler_registry: dict[str, Callable[[Task], Any]] or None
        :param prompt_handler: Callback to request permission for a task.
        :type prompt_handler: Callable[[Task], bool] or None
        :return: List of handler results.
        :rtype: list[Any]
        :raises RuntimeError: If a task is blocked or no handler exists.
        """
        if mode not in {"prompt", "auto", "dry_run"}:
            raise ValueError(f"Unsupported plan execution mode: {mode}")
        handler_registry = handler_registry or {}
        results: List[Any] = []
        self.missing_handlers = []

        for task in self.tasks:
            if task.status == "complete":
                continue
            if task.status == "blocked":
                raise RuntimeError(task.reason or f"Task '{task.name}' is blocked")
            if mode == "dry_run":
                continue
            if mode == "prompt":
                if prompt_handler is None or not prompt_handler(task):
                    raise RuntimeError(f"Task '{task.name}' declined by prompt handler")
            handler = handler_registry.get(task.kind)
            if handler is None:
                task.status = "blocked"
                task.reason = f"No handler registered for task kind '{task.kind}'"
                self.missing_handlers.append(task.kind)
                self.status = "blocked"
                raise RuntimeError(task.reason)
            results.append(handler(task))
            task.status = "complete"

        self.status = _plan_status(self.tasks)
        return results


def normalize_task_kind(name: str) -> str:
    """
    Normalize a task name into the canonical task kind.

    :param name: Task name or alias.
    :type name: str
    :return: Canonical task kind.
    :rtype: str
    """
    if not name:
        return name
    lower = name.strip().lower()
    return TASK_KIND_ALIASES.get(lower, lower)


def _default_pipeline_config() -> Dict[str, Any]:
    return {
        "stages": [
            {
                "extractor_id": "pass-through-text",
                "configuration": {},
            }
        ]
    }


def _pipeline_configuration_id(pipeline_config: Dict[str, Any]) -> str:
    manifest = create_extraction_configuration_manifest(
        extractor_id="pipeline",
        name="default",
        configuration=pipeline_config,
    )
    return manifest.configuration_id


def _catalog_generated_at(corpus: Corpus) -> str:
    return corpus.catalog_generated_at()


def _list_retrieval_snapshots(corpus: Corpus) -> List[RetrievalSnapshot]:
    snapshots: List[RetrievalSnapshot] = []
    if not corpus.retrieval_dir.is_dir():
        return snapshots
    for retriever_dir in sorted(corpus.retrieval_dir.iterdir()):
        if not retriever_dir.is_dir():
            continue
        for snapshot_dir in sorted(retriever_dir.iterdir()):
            manifest_path = snapshot_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                snapshots.append(corpus.load_snapshot(snapshot_dir.name))
            except Exception:
                continue
    return snapshots


def _find_retrieval_snapshot(
    *,
    corpus: Corpus,
    retriever_id: str,
    configuration_id: str,
    catalog_generated_at: str,
) -> Optional[RetrievalSnapshot]:
    for snapshot in _list_retrieval_snapshots(corpus):
        if snapshot.configuration.retriever_id != retriever_id:
            continue
        if snapshot.configuration.configuration_id != configuration_id:
            continue
        if snapshot.catalog_generated_at != catalog_generated_at:
            continue
        return snapshot
    return None


def _find_extraction_snapshot(
    *,
    corpus: Corpus,
    configuration_id: str,
    catalog_generated_at: str,
) -> Optional[ExtractionSnapshotListEntry]:
    entries = corpus.list_extraction_snapshots(extractor_id="pipeline")
    for entry in entries:
        if entry.configuration_id != configuration_id:
            continue
        if entry.catalog_generated_at != catalog_generated_at:
            continue
        return entry
    return None


def _build_task(
    *,
    name: str,
    kind: str,
    target_type: str,
    target_id: str,
    depends_on: Optional[List[Task]] = None,
    status: str = "ready",
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Task:
    return Task(
        name=name,
        kind=kind,
        target_type=target_type,
        target_id=target_id,
        depends_on=list(depends_on or []),
        status=status,
        reason=reason,
        metadata=dict(metadata or {}),
    )


def _flatten_tasks(task: Task, seen: Optional[set[int]] = None) -> List[Task]:
    seen = seen or set()
    tasks: List[Task] = []
    for child in task.depends_on:
        tasks.extend(_flatten_tasks(child, seen))
    if id(task) not in seen:
        tasks.append(task)
        seen.add(id(task))
    return tasks


def _plan_status(tasks: List[Task]) -> str:
    if any(task.status == "blocked" for task in tasks):
        return "blocked"
    if all(task.status == "complete" for task in tasks):
        return "complete"
    return "ready"


def build_default_handler_registry(
    corpus: Corpus,
) -> Dict[str, Callable[[Task], Any]]:
    """
    Build default task handlers for extraction and indexing.

    :param corpus: Corpus to operate on.
    :type corpus: Corpus
    :return: Mapping of task kinds to handler functions.
    :rtype: dict[str, Callable[[Task], Any]]
    """

    def _handle_extract(task: Task) -> Any:
        pipeline_config = task.metadata.get("pipeline") or _default_pipeline_config()
        force = bool(task.metadata.get("force", False))
        max_workers = int(task.metadata.get("max_workers", 1))
        return build_extraction_snapshot(
            corpus,
            extractor_id="pipeline",
            configuration_name="default",
            configuration=pipeline_config,
            force=force,
            max_workers=max_workers,
        )

    def _handle_index(task: Task) -> Any:
        retriever = get_retriever(task.target_id)
        index_config = task.metadata.get("index_config") or {}
        return retriever.build_snapshot(
            corpus, configuration_name="index", configuration=index_config
        )

    return {
        "extract": _handle_extract,
        "index": _handle_index,
    }


def build_plan_for_extract(
    corpus: Corpus,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
    load_handler_available: bool = False,
    force: bool = False,
    max_workers: int = 1,
) -> Plan:
    """
    Build a dependency plan for corpus extraction.

    :param corpus: Corpus to plan against.
    :type corpus: Corpus
    :param pipeline_config: Optional extraction pipeline configuration.
    :type pipeline_config: dict[str, Any] or None
    :param load_handler_available: Whether a load handler is available.
    :type load_handler_available: bool
    :param force: Whether to force extraction even when snapshots exist.
    :type force: bool
    :param max_workers: Maximum number of concurrent extraction workers.
    :type max_workers: int
    :return: Planned task graph for extraction.
    :rtype: Plan
    """
    pipeline_config = pipeline_config or _default_pipeline_config()
    catalog_generated_at = _catalog_generated_at(corpus)
    configuration_id = _pipeline_configuration_id(pipeline_config)
    snapshot = _find_extraction_snapshot(
        corpus=corpus,
        configuration_id=configuration_id,
        catalog_generated_at=catalog_generated_at,
    )
    has_items = corpus.has_items()

    extract_status = "complete" if snapshot and not force else "ready"
    extract_task = _build_task(
        name="extract",
        kind="extract",
        target_type="corpus",
        target_id=corpus.uri,
        status=extract_status,
        metadata={
            "pipeline": pipeline_config,
            "force": force,
            "max_workers": max_workers,
        },
    )

    if not has_items:
        if load_handler_available:
            load_task = _build_task(
                name="load",
                kind="load",
                target_type="corpus",
                target_id=corpus.uri,
                status="ready",
            )
            extract_task.depends_on.append(load_task)
        else:
            extract_task.status = "complete"
            extract_task.reason = None

    tasks = _flatten_tasks(extract_task)
    return Plan(tasks=tasks, root=extract_task, status=_plan_status(tasks))


def build_plan_for_index(
    corpus: Corpus,
    retriever_id: str,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
    index_config: Optional[Dict[str, Any]] = None,
    load_handler_available: bool = False,
) -> Plan:
    """
    Build a dependency plan for retriever indexing.

    :param corpus: Corpus to plan against.
    :type corpus: Corpus
    :param retriever_id: Retriever identifier.
    :type retriever_id: str
    :param pipeline_config: Optional extraction pipeline configuration.
    :type pipeline_config: dict[str, Any] or None
    :param index_config: Optional retriever index configuration.
    :type index_config: dict[str, Any] or None
    :param load_handler_available: Whether a load handler is available.
    :type load_handler_available: bool
    :return: Planned task graph for indexing.
    :rtype: Plan
    """
    pipeline_config = pipeline_config or _default_pipeline_config()
    index_config = index_config or {}
    catalog_generated_at = _catalog_generated_at(corpus)

    configuration_manifest = create_configuration_manifest(
        retriever_id=retriever_id,
        name="index",
        configuration=index_config,
    )
    snapshot = _find_retrieval_snapshot(
        corpus=corpus,
        retriever_id=retriever_id,
        configuration_id=configuration_manifest.configuration_id,
        catalog_generated_at=catalog_generated_at,
    )
    index_status = "complete" if snapshot else "ready"
    index_task = _build_task(
        name="index",
        kind="index",
        target_type="retriever",
        target_id=retriever_id,
        status=index_status,
        metadata={"index_config": index_config},
    )

    extract_plan = build_plan_for_extract(
        corpus,
        pipeline_config=pipeline_config,
        load_handler_available=load_handler_available,
    )
    index_task.depends_on.append(extract_plan.root)
    if extract_plan.root.status == "blocked":
        index_task.status = "blocked"
        index_task.reason = extract_plan.root.reason
    tasks = _flatten_tasks(index_task)
    return Plan(tasks=tasks, root=index_task, status=_plan_status(tasks))


def build_plan_for_query(
    corpus: Corpus,
    retriever_id: str,
    *,
    pipeline_config: Optional[Dict[str, Any]] = None,
    index_config: Optional[Dict[str, Any]] = None,
    load_handler_available: bool = False,
) -> Plan:
    """
    Build a dependency plan for retrieval queries.

    :param corpus: Corpus to plan against.
    :type corpus: Corpus
    :param retriever_id: Retriever identifier.
    :type retriever_id: str
    :param pipeline_config: Optional extraction pipeline configuration.
    :type pipeline_config: dict[str, Any] or None
    :param index_config: Optional retriever index configuration.
    :type index_config: dict[str, Any] or None
    :param load_handler_available: Whether a load handler is available.
    :type load_handler_available: bool
    :return: Planned task graph for querying.
    :rtype: Plan
    """
    index_plan = build_plan_for_index(
        corpus,
        retriever_id,
        pipeline_config=pipeline_config,
        index_config=index_config,
        load_handler_available=load_handler_available,
    )
    query_task = _build_task(
        name="query",
        kind="query",
        target_type="retriever",
        target_id=retriever_id,
        status="ready",
    )
    if index_plan.root.status != "complete":
        query_task.depends_on.append(index_plan.root)
    if index_plan.root.status == "blocked":
        query_task.status = "blocked"
        query_task.reason = index_plan.root.reason
    tasks = _flatten_tasks(query_task)
    return Plan(tasks=tasks, root=query_task, status=_plan_status(tasks))


def build_plan_for_load(corpus: Corpus, *, handler_available: bool = False) -> Plan:
    """
    Build a dependency plan for corpus loading.

    :param corpus: Corpus to plan against.
    :type corpus: Corpus
    :param handler_available: Whether a load handler is available.
    :type handler_available: bool
    :return: Planned task graph for loading.
    :rtype: Plan
    """
    has_items = corpus.has_items()
    status = "complete" if has_items else "ready" if handler_available else "blocked"
    reason = None
    if status == "blocked":
        reason = "Corpus has no items and no load handler is available"
    load_task = _build_task(
        name="load",
        kind="load",
        target_type="corpus",
        target_id=corpus.uri,
        status=status,
        reason=reason,
    )
    tasks = [load_task]
    return Plan(tasks=tasks, root=load_task, status=_plan_status(tasks))
