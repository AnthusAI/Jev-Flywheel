"""
Text extraction snapshots for Biblicus.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .errors import ExtractionSnapshotFatalError
from .extractors import get_extractor
from .models import CatalogItem, ExtractionStageOutput
from .time import utc_now_iso

Corpus = Any


def _hash_text(text: str) -> str:
    """
    Hash text for deterministic extraction identifiers.

    :param text: Text payload.
    :type text: str
    :return: Secure Hash Algorithm 256 hex digest.
    :rtype: str
    """
    return sha256(text.encode("utf-8")).hexdigest()


class ExtractionConfigurationManifest(BaseModel):
    """
    Reproducible configuration for an extraction plugin snapshot.

    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar extractor_id: Extractor plugin identifier.
    :vartype extractor_id: str
    :ivar name: Human-readable configuration name.
    :vartype name: str
    :ivar created_at: International Organization for Standardization 8601 timestamp.
    :vartype created_at: str
    :ivar configuration: Extractor-specific configuration values.
    :vartype configuration: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    configuration_id: str
    extractor_id: str
    name: str
    created_at: str
    configuration: Dict[str, Any] = Field(default_factory=dict)


class ExtractionStageResult(BaseModel):
    """
    Per-item result record for a single pipeline stage.

    :ivar stage_index: One-based pipeline stage index.
    :vartype stage_index: int
    :ivar extractor_id: Extractor identifier for the stage.
    :vartype extractor_id: str
    :ivar status: Stage status, extracted, skipped, or errored.
    :vartype status: str
    :ivar text_relpath: Relative path to the stage text artifact, when extracted.
    :vartype text_relpath: str or None
    :ivar text_characters: Character count of the extracted text.
    :vartype text_characters: int
    :ivar producer_extractor_id: Extractor identifier that produced the text content.
    :vartype producer_extractor_id: str or None
    :ivar source_stage_index: Optional stage index that supplied the text for selection-style extractors.
    :vartype source_stage_index: int or None
    :ivar confidence: Optional confidence score from 0.0 to 1.0.
    :vartype confidence: float or None
    :ivar metadata_relpath: Relative path to the stage metadata artifact, when present.
    :vartype metadata_relpath: str or None
    :ivar error_type: Optional error type name for errored stages.
    :vartype error_type: str or None
    :ivar error_message: Optional error message for errored stages.
    :vartype error_message: str or None
    """

    model_config = ConfigDict(extra="forbid")

    stage_index: int = Field(ge=1)
    extractor_id: str
    status: str
    text_relpath: Optional[str] = None
    text_characters: int = Field(default=0, ge=0)
    producer_extractor_id: Optional[str] = None
    source_stage_index: Optional[int] = Field(default=None, ge=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata_relpath: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ExtractionItemResult(BaseModel):
    """
    Per-item result record for an extraction snapshot.

    :ivar item_id: Item identifier.
    :vartype item_id: str
    :ivar status: Final result status, extracted, skipped, or errored.
    :vartype status: str
    :ivar final_text_relpath: Relative path to the final extracted text artifact, when extracted.
    :vartype final_text_relpath: str or None
    :ivar final_metadata_relpath: Relative path to the final metadata artifact, when present.
    :vartype final_metadata_relpath: str or None
    :ivar final_stage_index: Pipeline stage index that produced the final text.
    :vartype final_stage_index: int or None
    :ivar final_stage_extractor_id: Extractor identifier of the stage that produced the final text.
    :vartype final_stage_extractor_id: str or None
    :ivar final_producer_extractor_id: Extractor identifier that produced the final text content.
    :vartype final_producer_extractor_id: str or None
    :ivar final_source_stage_index: Optional stage index that supplied the final text for selection-style extractors.
    :vartype final_source_stage_index: int or None
    :ivar error_type: Optional error type name when no extracted text was produced.
    :vartype error_type: str or None
    :ivar error_message: Optional error message when no extracted text was produced.
    :vartype error_message: str or None
    :ivar stage_results: Per-stage results recorded for this item.
    :vartype stage_results: list[ExtractionStageResult]
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    status: str
    final_text_relpath: Optional[str] = None
    final_metadata_relpath: Optional[str] = None
    final_stage_index: Optional[int] = Field(default=None, ge=1)
    final_stage_extractor_id: Optional[str] = None
    final_producer_extractor_id: Optional[str] = None
    final_source_stage_index: Optional[int] = Field(default=None, ge=1)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stage_results: List[ExtractionStageResult] = Field(default_factory=list)


class ExtractionSnapshotManifest(BaseModel):
    """
    Immutable record describing an extraction snapshot.

    :ivar snapshot_id: Unique snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration: Configuration manifest for this snapshot.
    :vartype configuration: ExtractionConfigurationManifest
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar catalog_generated_at: Catalog timestamp used for the snapshot.
    :vartype catalog_generated_at: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for snapshot creation.
    :vartype created_at: str
    :ivar items: Per-item results.
    :vartype items: list[ExtractionItemResult]
    :ivar stats: Snapshot statistics.
    :vartype stats: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    configuration: ExtractionConfigurationManifest
    corpus_uri: str
    catalog_generated_at: str
    created_at: str
    items: List[ExtractionItemResult] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


def create_extraction_configuration_manifest(
    *, extractor_id: str, name: str, configuration: Dict[str, Any]
) -> ExtractionConfigurationManifest:
    """
    Create a deterministic extraction configuration manifest.

    :param extractor_id: Extractor plugin identifier.
    :type extractor_id: str
    :param name: Human configuration name.
    :type name: str
    :param configuration: Extractor configuration.
    :type configuration: dict[str, Any]
    :return: Configuration manifest.
    :rtype: ExtractionConfigurationManifest
    """
    configuration_payload = json.dumps(
        {"extractor_id": extractor_id, "name": name, "configuration": configuration},
        sort_keys=True,
    )
    configuration_id = _hash_text(configuration_payload)
    return ExtractionConfigurationManifest(
        configuration_id=configuration_id,
        extractor_id=extractor_id,
        name=name,
        created_at=utc_now_iso(),
        configuration=configuration,
    )


def create_extraction_snapshot_manifest(
    corpus: "Corpus", *, configuration: ExtractionConfigurationManifest
) -> ExtractionSnapshotManifest:
    """
    Create a new extraction snapshot manifest for a corpus.

    :param corpus: Corpus associated with the snapshot.
    :type corpus: Corpus
    :param configuration: Configuration manifest.
    :type configuration: ExtractionConfigurationManifest
    :return: Snapshot manifest.
    :rtype: ExtractionSnapshotManifest
    """
    catalog = corpus.load_catalog()
    snapshot_id = _hash_text(f"{configuration.configuration_id}:{catalog.generated_at}")
    return ExtractionSnapshotManifest(
        snapshot_id=snapshot_id,
        configuration=configuration,
        corpus_uri=corpus.uri,
        catalog_generated_at=catalog.generated_at,
        created_at=utc_now_iso(),
        items=[],
        stats={},
    )


def write_extraction_snapshot_manifest(
    *, snapshot_dir: Path, manifest: ExtractionSnapshotManifest
) -> None:
    """
    Persist an extraction snapshot manifest to a snapshot directory.

    :param snapshot_dir: Extraction snapshot directory.
    :type snapshot_dir: Path
    :param manifest: Snapshot manifest to write.
    :type manifest: ExtractionSnapshotManifest
    :return: None.
    :rtype: None
    """
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_extraction_latest_pointer(
    *, extractor_dir: Path, manifest: ExtractionSnapshotManifest
) -> None:
    """
    Persist the latest pointer for an extractor.

    :param extractor_dir: Extractor directory containing snapshots.
    :type extractor_dir: Path
    :param manifest: Snapshot manifest used for the pointer.
    :type manifest: ExtractionSnapshotManifest
    :return: None.
    :rtype: None
    """
    latest_path = extractor_dir / "latest.json"
    latest_path.write_text(
        json.dumps(
            {"snapshot_id": manifest.snapshot_id, "created_at": manifest.created_at},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _ensure_extraction_alias_snapshot_dir(
    *,
    corpus: "Corpus",
    stage_extractor_id: str,
    manifest: ExtractionSnapshotManifest,
) -> Path:
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=stage_extractor_id, snapshot_id=manifest.snapshot_id
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_extraction_snapshot_manifest(snapshot_dir=snapshot_dir, manifest=manifest)
    write_extraction_latest_pointer(extractor_dir=snapshot_dir.parent, manifest=manifest)
    alias_path = snapshot_dir / "alias.json"
    alias_path.write_text(
        json.dumps(
            {
                "source_extractor_id": manifest.configuration.extractor_id,
                "source_snapshot_id": manifest.snapshot_id,
                "stage_extractor_id": stage_extractor_id,
                "created_at": manifest.created_at,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


def _write_alias_text_artifact(*, alias_snapshot_dir: Path, item: CatalogItem, text: str) -> str:
    text_dir = alias_snapshot_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("text") / f"{item.id}.txt")
    (alias_snapshot_dir / relpath).write_text(text, encoding="utf-8")
    return relpath


def _write_alias_metadata_artifact(
    *, alias_snapshot_dir: Path, item: CatalogItem, metadata: Dict[str, Any]
) -> Optional[str]:
    if not metadata:
        return None
    metadata_dir = alias_snapshot_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("metadata") / f"{item.id}.json")
    (alias_snapshot_dir / relpath).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return relpath


def write_extracted_text_artifact(*, snapshot_dir: Path, item: CatalogItem, text: str) -> str:
    """
    Write an extracted text artifact for an item into the snapshot directory.

    :param snapshot_dir: Extraction snapshot directory.
    :type snapshot_dir: Path
    :param item: Catalog item being extracted.
    :type item: CatalogItem
    :param text: Extracted text.
    :type text: str
    :return: Relative path to the stored text artifact.
    :rtype: str
    """
    text_dir = snapshot_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("text") / f"{item.id}.txt")
    path = snapshot_dir / relpath
    path.write_text(text, encoding="utf-8")
    return relpath


def _pipeline_stage_dir_name(*, stage_index: int, extractor_id: str) -> str:
    """
    Build a stable directory name for a pipeline stage.

    :param stage_index: One-based pipeline stage index.
    :type stage_index: int
    :param extractor_id: Extractor identifier for the stage.
    :type extractor_id: str
    :return: Directory name for the stage.
    :rtype: str
    """
    return f"{stage_index:02d}-{extractor_id}"


def write_pipeline_stage_text_artifact(
    *,
    snapshot_dir: Path,
    stage_index: int,
    extractor_id: str,
    item: CatalogItem,
    text: str,
) -> str:
    """
    Write a pipeline stage text artifact for an item.

    :param snapshot_dir: Extraction snapshot directory.
    :type snapshot_dir: Path
    :param stage_index: One-based pipeline stage index.
    :type stage_index: int
    :param extractor_id: Extractor identifier for the stage.
    :type extractor_id: str
    :param item: Catalog item being extracted.
    :type item: CatalogItem
    :param text: Extracted text content.
    :type text: str
    :return: Relative path to the stored stage text artifact.
    :rtype: str
    """
    stage_dir_name = _pipeline_stage_dir_name(stage_index=stage_index, extractor_id=extractor_id)
    text_dir = snapshot_dir / "stages" / stage_dir_name / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("stages") / stage_dir_name / "text" / f"{item.id}.txt")
    (snapshot_dir / relpath).write_text(text, encoding="utf-8")
    return relpath


def write_extracted_metadata_artifact(
    *, snapshot_dir: Path, item: CatalogItem, metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Write an extracted metadata artifact for an item into the snapshot directory.

    :param snapshot_dir: Extraction snapshot directory.
    :type snapshot_dir: Path
    :param item: Catalog item being extracted.
    :type item: CatalogItem
    :param metadata: Metadata dictionary to persist.
    :type metadata: dict[str, Any]
    :return: Relative path to the stored metadata artifact, or None if empty.
    :rtype: str or None
    """
    if not metadata:
        return None
    metadata_dir = snapshot_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("metadata") / f"{item.id}.json")
    path = snapshot_dir / relpath
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return relpath


def write_pipeline_stage_metadata_artifact(
    *,
    snapshot_dir: Path,
    stage_index: int,
    extractor_id: str,
    item: CatalogItem,
    metadata: Dict[str, Any],
) -> Optional[str]:
    """
    Write a pipeline stage metadata artifact for an item.

    :param snapshot_dir: Extraction snapshot directory.
    :type snapshot_dir: Path
    :param stage_index: One-based pipeline stage index.
    :type stage_index: int
    :param extractor_id: Extractor identifier for the stage.
    :type extractor_id: str
    :param item: Catalog item being extracted.
    :type item: CatalogItem
    :param metadata: Metadata dictionary to persist.
    :type metadata: dict[str, Any]
    :return: Relative path to the stored stage metadata artifact, or None if empty.
    :rtype: str or None
    """
    if not metadata:
        return None
    stage_dir_name = _pipeline_stage_dir_name(stage_index=stage_index, extractor_id=extractor_id)
    metadata_dir = snapshot_dir / "stages" / stage_dir_name / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    relpath = str(Path("stages") / stage_dir_name / "metadata" / f"{item.id}.json")
    (snapshot_dir / relpath).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return relpath


def _final_output_from_stages(
    stage_outputs: List[ExtractionStageOutput],
) -> Optional[ExtractionStageOutput]:
    """
    Select the final pipeline output for an item.

    The final output is the last extracted stage output in pipeline order.

    :param stage_outputs: Extracted outputs produced by pipeline stages.
    :type stage_outputs: list[biblicus.models.ExtractionStageOutput]
    :return: Final stage output or None when no stages produced extracted text.
    :rtype: biblicus.models.ExtractionStageOutput or None
    """
    if not stage_outputs:
        return None
    return stage_outputs[-1]


def build_extraction_snapshot(
    corpus: "Corpus",
    *,
    extractor_id: str,
    configuration_name: str,
    configuration: Dict[str, Any],
    force: bool = False,
    max_workers: int = 1,
) -> ExtractionSnapshotManifest:
    """
    Build an extraction snapshot for a corpus using the pipeline extractor.

    :param corpus: Corpus to extract from.
    :type corpus: Corpus
    :param extractor_id: Extractor plugin identifier (must be ``pipeline``).
    :type extractor_id: str
    :param configuration_name: Human-readable configuration name.
    :type configuration_name: str
    :param configuration: Extractor configuration mapping.
    :type configuration: dict[str, Any]
    :param force: Whether to reprocess items even if artifacts already exist.
    :type force: bool
    :param max_workers: Maximum number of concurrent workers.
    :type max_workers: int
    :return: Extraction snapshot manifest describing the build.
    :rtype: ExtractionSnapshotManifest
    :raises KeyError: If the extractor identifier is unknown.
    :raises ValueError: If the extractor configuration is invalid.
    :raises OSError: If the snapshot directory or artifacts cannot be written.
    :raises ExtractionSnapshotFatalError: If the extractor is not the pipeline.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    extractor = get_extractor(extractor_id)
    parsed_config = extractor.validate_config(configuration)
    config_manifest = create_extraction_configuration_manifest(
        extractor_id=extractor_id,
        name=configuration_name,
        configuration=parsed_config.model_dump(),
    )
    manifest = create_extraction_snapshot_manifest(corpus, configuration=config_manifest)
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=extractor_id, snapshot_id=manifest.snapshot_id
    )
    if snapshot_dir.exists():
        try:
            manifest = corpus.load_extraction_snapshot_manifest(
                extractor_id=extractor_id, snapshot_id=manifest.snapshot_id
            )
        except FileNotFoundError:
            manifest_path = snapshot_dir / "manifest.json"
            if manifest_path.exists():
                raise
    else:
        snapshot_dir.mkdir(parents=True, exist_ok=False)

    catalog = corpus.load_catalog()
    if extractor_id != "pipeline":
        raise ExtractionSnapshotFatalError("Extraction snapshots must use the pipeline extractor")

    parsed_config_data = parsed_config.model_dump()
    pipeline_stages = parsed_config_data.get("stages")
    if not isinstance(pipeline_stages, list) or not pipeline_stages:
        raise ValueError("Pipeline configuration must contain at least one stage")

    validated_stages: List[Tuple[str, BaseModel]] = []
    for stage_data in pipeline_stages:
        if not isinstance(stage_data, dict):
            raise ValueError("Pipeline stage entries must be objects")
        stage_extractor_id = str(stage_data.get("extractor_id", "")).strip()
        if not stage_extractor_id:
            raise ValueError("Pipeline stage missing extractor_id")
        stage_configuration = stage_data.get("configuration", {})
        if not isinstance(stage_configuration, dict):
            raise ValueError("Pipeline stage configuration must be an object")
        stage_extractor = get_extractor(stage_extractor_id)
        parsed_stage_config = stage_extractor.validate_config(stage_configuration)
        validated_stages.append((stage_extractor_id, parsed_stage_config))

    previous_items = {item.item_id: item for item in (manifest.items or [])}
    extracted_items: List[ExtractionItemResult] = []
    extracted_count = 0
    skipped_count = 0
    errored_count = 0
    extracted_nonempty_count = 0
    extracted_empty_count = 0
    already_text_item_count = 0
    needs_extraction_item_count = 0
    converted_item_count = 0
    total_item_count = len(catalog.items)
    if total_item_count <= 25:
        log_interval = 1
    elif total_item_count <= 100:
        log_interval = 10
    else:
        log_interval = 25
    start_time = time.perf_counter()
    processed_count = 0
    print(
        f"[extract] building snapshot {manifest.snapshot_id} items={total_item_count} "
        f"workers={max_workers}",
        flush=True,
        file=sys.stderr,
    )

    def _write_partial_manifest() -> None:
        stats = {
            "total_items": total_item_count,
            "already_text_items": already_text_item_count,
            "needs_extraction_items": needs_extraction_item_count,
            "extracted_items": extracted_count,
            "extracted_nonempty_items": extracted_nonempty_count,
            "extracted_empty_items": extracted_empty_count,
            "skipped_items": skipped_count,
            "errored_items": errored_count,
            "converted_items": converted_item_count,
        }
        partial_manifest = manifest.model_copy(update={"items": extracted_items, "stats": stats})
        write_extraction_snapshot_manifest(snapshot_dir=snapshot_dir, manifest=partial_manifest)

    lock = threading.Lock()
    progress_lock = threading.Lock()
    current_item_id: Optional[str] = None
    current_stage_label: Optional[str] = None
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(30):
            with progress_lock:
                active_item = current_item_id
                active_stage = current_stage_label
                processed = processed_count
            elapsed = time.perf_counter() - start_time
            print(
                f"[extract] heartbeat processed={processed}/{total_item_count} "
                f"active_item={active_item or 'none'} stage={active_stage or 'none'} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
                file=sys.stderr,
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    def _load_stage_cache(
        *, stage_index: int, extractor_id: str, item: CatalogItem
    ) -> Optional[Tuple[ExtractionStageResult, ExtractionStageOutput]]:
        stage_dir_name = _pipeline_stage_dir_name(
            stage_index=stage_index, extractor_id=extractor_id
        )
        text_relpath = str(Path("stages") / stage_dir_name / "text" / f"{item.id}.txt")
        text_path = snapshot_dir / text_relpath
        if not text_path.is_file():
            legacy_text_path = next(
                (
                    path
                    for path in snapshot_dir.parent.glob(
                        f"*/stages/{stage_dir_name}/text/{item.id}.txt"
                    )
                    if path.is_file()
                ),
                None,
            )
            if legacy_text_path is None:
                return None
            text_value = legacy_text_path.read_text(encoding="utf-8")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text_value, encoding="utf-8")
        else:
            text_value = text_path.read_text(encoding="utf-8")
        metadata_relpath = str(Path("stages") / stage_dir_name / "metadata" / f"{item.id}.json")
        metadata_path = snapshot_dir / metadata_relpath
        metadata_value: Dict[str, Any] = {}
        if metadata_path.is_file():
            metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            legacy_metadata_path = next(
                (
                    path
                    for path in snapshot_dir.parent.glob(
                        f"*/stages/{stage_dir_name}/metadata/{item.id}.json"
                    )
                    if path.is_file()
                ),
                None,
            )
            if legacy_metadata_path is not None:
                metadata_value = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(
                    legacy_metadata_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
        stage_result = ExtractionStageResult(
            stage_index=stage_index,
            extractor_id=extractor_id,
            status="extracted",
            text_relpath=text_relpath,
            text_characters=len(text_value),
            producer_extractor_id=extractor_id,
            source_stage_index=None,
            confidence=None,
            metadata_relpath=metadata_relpath if metadata_path.is_file() else None,
            error_type=None,
            error_message=None,
        )
        stage_output = ExtractionStageOutput(
            stage_index=stage_index,
            extractor_id=extractor_id,
            status="extracted",
            text=text_value,
            text_characters=len(text_value),
            producer_extractor_id=extractor_id,
            source_stage_index=None,
            confidence=None,
            metadata=metadata_value,
            error_type=None,
            error_message=None,
        )
        return stage_result, stage_output

    def _build_item_result(item: CatalogItem) -> Tuple[ExtractionItemResult, Dict[str, int]]:
        nonlocal current_item_id
        nonlocal current_stage_label
        media_type = item.media_type
        item_is_text = media_type == "text/markdown" or media_type.startswith("text/")
        stats_delta = {
            "already_text_items": 1 if item_is_text else 0,
            "needs_extraction_items": 0 if item_is_text else 1,
            "extracted_items": 0,
            "extracted_nonempty_items": 0,
            "extracted_empty_items": 0,
            "skipped_items": 0,
            "errored_items": 0,
            "converted_items": 0,
        }

        with progress_lock:
            current_item_id = item.id
            current_stage_label = "prepare"

        final_text_relpath = str(Path("text") / f"{item.id}.txt")
        final_metadata_relpath = str(Path("metadata") / f"{item.id}.json")
        final_text_path = snapshot_dir / final_text_relpath

        if not force and not final_text_path.is_file():
            legacy_text_path = next(
                (
                    path
                    for path in snapshot_dir.parent.glob(f"*/text/{item.id}.txt")
                    if path.is_file()
                ),
                None,
            )
            if legacy_text_path is not None:
                legacy_text = legacy_text_path.read_text(encoding="utf-8")
                final_text_path.parent.mkdir(parents=True, exist_ok=True)
                final_text_path.write_text(legacy_text, encoding="utf-8")
                legacy_metadata_path = (
                    legacy_text_path.parent.parent / "metadata" / f"{item.id}.json"
                )
                if legacy_metadata_path.is_file():
                    legacy_metadata_dest = snapshot_dir / final_metadata_relpath
                    legacy_metadata_dest.parent.mkdir(parents=True, exist_ok=True)
                    legacy_metadata_dest.write_text(
                        legacy_metadata_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

        if not force and final_text_path.is_file():
            final_text_value = final_text_path.read_text(encoding="utf-8")
            cached_item = previous_items.get(item.id)
            if cached_item and cached_item.final_stage_extractor_id:
                alias_snapshot_dir = _ensure_extraction_alias_snapshot_dir(
                    corpus=corpus,
                    stage_extractor_id=cached_item.final_stage_extractor_id,
                    manifest=manifest,
                )
                _write_alias_text_artifact(
                    alias_snapshot_dir=alias_snapshot_dir,
                    item=item,
                    text=final_text_value,
                )
                metadata_value: Dict[str, Any] = {}
                metadata_path = snapshot_dir / final_metadata_relpath
                if metadata_path.is_file():
                    metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
                _write_alias_metadata_artifact(
                    alias_snapshot_dir=alias_snapshot_dir,
                    item=item,
                    metadata=metadata_value,
                )
            stats_delta["extracted_items"] = 1
            if final_text_value.strip():
                stats_delta["extracted_nonempty_items"] = 1
                if not item_is_text:
                    stats_delta["converted_items"] = 1
            else:
                stats_delta["extracted_empty_items"] = 1
            if cached_item is not None:
                return cached_item, stats_delta
            return (
                ExtractionItemResult(
                    item_id=item.id,
                    status="extracted",
                    final_text_relpath=final_text_relpath,
                    final_metadata_relpath=(
                        final_metadata_relpath
                        if (snapshot_dir / final_metadata_relpath).is_file()
                        else None
                    ),
                    final_stage_index=None,
                    final_stage_extractor_id=None,
                    final_producer_extractor_id=None,
                    final_source_stage_index=None,
                    error_type=None,
                    error_message=None,
                    stage_results=[],
                ),
                stats_delta,
            )

        stage_results: List[ExtractionStageResult] = []
        stage_outputs: List[ExtractionStageOutput] = []
        last_error_type: Optional[str] = None
        last_error_message: Optional[str] = None

        for stage_index, (stage_extractor_id, parsed_stage_config) in enumerate(
            validated_stages, start=1
        ):
            with progress_lock:
                current_stage_label = f"{stage_extractor_id}:{stage_index}"
            if not force:
                cached = _load_stage_cache(
                    stage_index=stage_index,
                    extractor_id=stage_extractor_id,
                    item=item,
                )
                if cached:
                    with progress_lock:
                        current_stage_label = f"{stage_extractor_id}:{stage_index}:cache"
                    cached_result, cached_output = cached
                    stage_results.append(cached_result)
                    stage_outputs.append(cached_output)
                    continue
            try:
                stage_extractor = get_extractor(stage_extractor_id)
                extracted_text = stage_extractor.extract_text(
                    corpus=corpus,
                    item=item,
                    config=parsed_stage_config,
                    previous_extractions=stage_outputs,
                )
            except Exception as extraction_error:
                if isinstance(extraction_error, ExtractionSnapshotFatalError):
                    raise
                last_error_type = extraction_error.__class__.__name__
                last_error_message = str(extraction_error)
                stage_results.append(
                    ExtractionStageResult(
                        stage_index=stage_index,
                        extractor_id=stage_extractor_id,
                        status="errored",
                        text_relpath=None,
                        text_characters=0,
                        producer_extractor_id=None,
                        source_stage_index=None,
                        error_type=last_error_type,
                        error_message=last_error_message,
                    )
                )
                continue

            if extracted_text is None:
                stage_results.append(
                    ExtractionStageResult(
                        stage_index=stage_index,
                        extractor_id=stage_extractor_id,
                        status="skipped",
                        text_relpath=None,
                        text_characters=0,
                        producer_extractor_id=None,
                        source_stage_index=None,
                        error_type=None,
                        error_message=None,
                    )
                )
                continue

            relpath = write_pipeline_stage_text_artifact(
                snapshot_dir=snapshot_dir,
                stage_index=stage_index,
                extractor_id=stage_extractor_id,
                item=item,
                text=extracted_text.text,
            )
            metadata_relpath = write_pipeline_stage_metadata_artifact(
                snapshot_dir=snapshot_dir,
                stage_index=stage_index,
                extractor_id=stage_extractor_id,
                item=item,
                metadata=extracted_text.metadata,
            )
            text_characters = len(extracted_text.text)
            stage_results.append(
                ExtractionStageResult(
                    stage_index=stage_index,
                    extractor_id=stage_extractor_id,
                    status="extracted",
                    text_relpath=relpath,
                    text_characters=text_characters,
                    producer_extractor_id=extracted_text.producer_extractor_id,
                    source_stage_index=extracted_text.source_stage_index,
                    confidence=extracted_text.confidence,
                    metadata_relpath=metadata_relpath,
                    error_type=None,
                    error_message=None,
                )
            )
            stage_outputs.append(
                ExtractionStageOutput(
                    stage_index=stage_index,
                    extractor_id=stage_extractor_id,
                    status="extracted",
                    text=extracted_text.text,
                    text_characters=text_characters,
                    producer_extractor_id=extracted_text.producer_extractor_id,
                    source_stage_index=extracted_text.source_stage_index,
                    confidence=extracted_text.confidence,
                    metadata=extracted_text.metadata,
                    error_type=None,
                    error_message=None,
                )
            )

        final_output = _final_output_from_stages(stage_outputs)
        if final_output is None:
            status = "errored" if last_error_type else "skipped"
            if status == "errored":
                stats_delta["errored_items"] = 1
            else:
                stats_delta["skipped_items"] = 1
            return (
                ExtractionItemResult(
                    item_id=item.id,
                    status=status,
                    final_text_relpath=None,
                    final_metadata_relpath=None,
                    final_stage_index=None,
                    final_stage_extractor_id=None,
                    final_producer_extractor_id=None,
                    final_source_stage_index=None,
                    error_type=last_error_type if status == "errored" else None,
                    error_message=last_error_message if status == "errored" else None,
                    stage_results=stage_results,
                ),
                stats_delta,
            )

        final_text = final_output.text or ""
        final_text_relpath = write_extracted_text_artifact(
            snapshot_dir=snapshot_dir, item=item, text=final_text
        )
        final_metadata_relpath = write_extracted_metadata_artifact(
            snapshot_dir=snapshot_dir, item=item, metadata=final_output.metadata
        )
        alias_snapshot_dir = _ensure_extraction_alias_snapshot_dir(
            corpus=corpus,
            stage_extractor_id=final_output.extractor_id,
            manifest=manifest,
        )
        _write_alias_text_artifact(
            alias_snapshot_dir=alias_snapshot_dir,
            item=item,
            text=final_text,
        )
        _write_alias_metadata_artifact(
            alias_snapshot_dir=alias_snapshot_dir,
            item=item,
            metadata=final_output.metadata,
        )
        stats_delta["extracted_items"] = 1
        if final_text.strip():
            stats_delta["extracted_nonempty_items"] = 1
            if not item_is_text:
                stats_delta["converted_items"] = 1
        else:
            stats_delta["extracted_empty_items"] = 1

        return (
            ExtractionItemResult(
                item_id=item.id,
                status="extracted",
                final_text_relpath=final_text_relpath,
                final_metadata_relpath=final_metadata_relpath,
                final_stage_index=final_output.stage_index,
                final_stage_extractor_id=final_output.extractor_id,
                final_producer_extractor_id=final_output.producer_extractor_id,
                final_source_stage_index=final_output.source_stage_index,
                error_type=None,
                error_message=None,
                stage_results=stage_results,
            ),
            stats_delta,
        )

    def _apply_result(item_result: ExtractionItemResult, stats_delta: Dict[str, int]) -> None:
        nonlocal extracted_count
        nonlocal skipped_count
        nonlocal errored_count
        nonlocal extracted_nonempty_count
        nonlocal extracted_empty_count
        nonlocal already_text_item_count
        nonlocal needs_extraction_item_count
        nonlocal converted_item_count
        nonlocal processed_count

        extracted_items.append(item_result)
        extracted_count += stats_delta["extracted_items"]
        skipped_count += stats_delta["skipped_items"]
        errored_count += stats_delta["errored_items"]
        extracted_nonempty_count += stats_delta["extracted_nonempty_items"]
        extracted_empty_count += stats_delta["extracted_empty_items"]
        already_text_item_count += stats_delta["already_text_items"]
        needs_extraction_item_count += stats_delta["needs_extraction_items"]
        converted_item_count += stats_delta["converted_items"]
        processed_count += 1
        if processed_count % log_interval == 0 or processed_count == total_item_count:
            elapsed = time.perf_counter() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[extract] processed {processed_count}/{total_item_count} "
                f"extracted={extracted_count} skipped={skipped_count} errored={errored_count} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
                file=sys.stderr,
            )
        _write_partial_manifest()

    def _process_and_record(item: CatalogItem) -> None:
        item_result, stats_delta = _build_item_result(item)
        with lock:
            _apply_result(item_result, stats_delta)

    if max_workers == 1:
        try:
            for item in catalog.items.values():
                _process_and_record(item)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1)
    else:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_build_item_result, item) for item in catalog.items.values()
                ]
                for future in as_completed(futures):
                    item_result, stats_delta = future.result()
                    with lock:
                        _apply_result(item_result, stats_delta)
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1)

    stats = {
        "total_items": total_item_count,
        "already_text_items": already_text_item_count,
        "needs_extraction_items": needs_extraction_item_count,
        "extracted_items": extracted_count,
        "extracted_nonempty_items": extracted_nonempty_count,
        "extracted_empty_items": extracted_empty_count,
        "skipped_items": skipped_count,
        "errored_items": errored_count,
        "converted_items": converted_item_count,
    }
    manifest = manifest.model_copy(update={"items": extracted_items, "stats": stats})
    write_extraction_snapshot_manifest(snapshot_dir=snapshot_dir, manifest=manifest)
    write_extraction_latest_pointer(extractor_dir=snapshot_dir.parent, manifest=manifest)

    # Auto-sync catalog to Amplify if configured
    if os.getenv("AMPLIFY_AUTO_SYNC_CATALOG", "false").lower() == "true":
        try:
            from .sync.amplify_publisher import AmplifyPublisher

            publisher = AmplifyPublisher(corpus.name)
            catalog_path = corpus.root / "catalog.json"

            if catalog_path.exists():
                result = publisher.sync_catalog(catalog_path, force=False)
                if not result.skipped:
                    print(
                        f"✓ Synced catalog: {result.created} created, {result.updated} updated, {result.deleted} deleted",
                        file=sys.stderr,
                    )
        except ImportError:
            # AmplifyPublisher not available, skip sync
            pass
        except Exception as e:
            # Don't fail extraction if sync fails
            print(f"Warning: Catalog sync failed: {e}", file=sys.stderr)

    return manifest


def load_or_build_extraction_snapshot(
    corpus: "Corpus",
    *,
    extractor_id: str,
    configuration_name: str,
    configuration: Dict[str, Any],
    max_workers: int = 1,
) -> ExtractionSnapshotManifest:
    """
    Load an extraction snapshot if it exists or build it when missing.

    :param corpus: Corpus to extract from.
    :type corpus: Corpus
    :param extractor_id: Extractor plugin identifier (must be ``pipeline``).
    :type extractor_id: str
    :param configuration_name: Human-readable configuration name.
    :type configuration_name: str
    :param configuration: Extractor configuration mapping.
    :type configuration: dict[str, Any]
    :param max_workers: Maximum number of concurrent workers.
    :type max_workers: int
    :return: Extraction snapshot manifest describing the build.
    :rtype: ExtractionSnapshotManifest
    """
    configuration_manifest = create_extraction_configuration_manifest(
        extractor_id=extractor_id,
        name=configuration_name,
        configuration=configuration,
    )
    snapshot_manifest = create_extraction_snapshot_manifest(
        corpus,
        configuration=configuration_manifest,
    )
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=extractor_id,
        snapshot_id=snapshot_manifest.snapshot_id,
    )
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.is_file():
        print(
            f"[extract] reusing snapshot {snapshot_manifest.snapshot_id}",
            flush=True,
            file=sys.stderr,
        )
        return corpus.load_extraction_snapshot_manifest(
            extractor_id=extractor_id,
            snapshot_id=snapshot_manifest.snapshot_id,
        )
    print(
        f"[extract] building snapshot {snapshot_manifest.snapshot_id}",
        flush=True,
        file=sys.stderr,
    )
    return build_extraction_snapshot(
        corpus,
        extractor_id=extractor_id,
        configuration_name=configuration_name,
        configuration=configuration,
        max_workers=max_workers,
    )
