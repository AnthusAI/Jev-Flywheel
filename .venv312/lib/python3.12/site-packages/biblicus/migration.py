"""
Corpus layout migration helpers for Biblicus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from .constants import (
    ANALYSIS_DIR_NAME,
    CORPUS_DIR_NAME,
    EXTRACTED_DIR_NAME,
    GRAPH_DIR_NAME,
    LEGACY_CORPUS_DIR_NAME,
    RETRIEVAL_DIR_NAME,
    SCHEMA_VERSION,
)
from .models import CorpusCatalog, CorpusConfig, RetrievalSnapshot
from .time import utc_now_iso


def migrate_layout(*, corpus_root: Path, force: bool = False) -> Dict[str, int]:
    """
    Migrate a legacy corpus layout to the current layout.

    :param corpus_root: Corpus root directory.
    :type corpus_root: Path
    :param force: Whether to overwrite collisions during migration.
    :type force: bool
    :return: Migration statistics.
    :rtype: dict[str, int]
    """
    root = corpus_root.resolve()
    old_meta = root / LEGACY_CORPUS_DIR_NAME
    new_meta = root / CORPUS_DIR_NAME
    stats = {
        "moved_raw_items": 0,
        "moved_extraction_snapshots": 0,
        "moved_graph_snapshots": 0,
        "moved_analysis_runs": 0,
        "moved_retrieval_snapshots": 0,
        "updated_catalog_items": 0,
        "updated_snapshot_artifacts": 0,
    }

    if not old_meta.is_dir():
        raise ValueError("Legacy corpus metadata folder (.biblicus) not found")

    if new_meta.exists():
        if not force:
            raise ValueError("metadata/ already exists; use --force to overwrite")
        shutil.rmtree(new_meta)

    shutil.move(str(old_meta), str(new_meta))

    _migrate_raw_items(root=root, meta_dir=new_meta, force=force, stats=stats)
    _migrate_snapshots(root=root, meta_dir=new_meta, force=force, stats=stats)
    _update_config_and_catalog(meta_dir=new_meta, stats=stats)

    return stats


def _migrate_raw_items(
    *, root: Path, meta_dir: Optional[Path] = None, force: bool, stats: Dict[str, int]
) -> None:
    meta_dir = meta_dir or (root / CORPUS_DIR_NAME)
    candidate_sources = [meta_dir / "raw", root / "raw"]
    for source in candidate_sources:
        if not source.is_dir():
            continue
        for entry in source.iterdir():
            dest = root / entry.name
            _move_entry(entry, dest, force=force)
            stats["moved_raw_items"] += 1
        shutil.rmtree(source)


def _migrate_snapshots(
    *, root: Path, meta_dir: Path, force: bool, stats: Dict[str, int]
) -> None:
    snapshots_root = meta_dir / "snapshots"
    if not snapshots_root.is_dir():
        return

    extraction_root = snapshots_root / "extraction"
    if extraction_root.is_dir():
        stats["moved_extraction_snapshots"] += _move_snapshot_tree(
            extraction_root, root / EXTRACTED_DIR_NAME, force=force
        )
        shutil.rmtree(extraction_root)

    graph_root = snapshots_root / "graph"
    if graph_root.is_dir():
        stats["moved_graph_snapshots"] += _move_snapshot_tree(
            graph_root, root / GRAPH_DIR_NAME, force=force
        )
        shutil.rmtree(graph_root)

    analysis_root = snapshots_root / "analysis"
    if analysis_root.is_dir():
        stats["moved_analysis_runs"] += _move_snapshot_tree(
            analysis_root, root / ANALYSIS_DIR_NAME, force=force
        )
        shutil.rmtree(analysis_root)

    evaluation_root = snapshots_root / "evaluation"
    if evaluation_root.is_dir():
        dest = root / ANALYSIS_DIR_NAME / "evaluation"
        _move_entry(evaluation_root, dest, force=force)

    stats["moved_retrieval_snapshots"] += _migrate_retrieval_snapshots(
        snapshots_root, root / RETRIEVAL_DIR_NAME, force=force, stats=stats
    )
    if snapshots_root.exists():
        shutil.rmtree(snapshots_root)

    _write_latest_pointers(root / EXTRACTED_DIR_NAME)
    _write_latest_pointers(root / GRAPH_DIR_NAME)
    _write_latest_pointers(root / ANALYSIS_DIR_NAME)
    _write_latest_pointers(root / RETRIEVAL_DIR_NAME)


def _move_snapshot_tree(source_root: Path, dest_root: Path, *, force: bool) -> int:
    moved = 0
    for extractor_dir in sorted(source_root.iterdir()):
        if not extractor_dir.is_dir():
            continue
        for snapshot_dir in sorted(extractor_dir.iterdir()):
            if not snapshot_dir.is_dir():
                continue
            dest = dest_root / extractor_dir.name / snapshot_dir.name
            _move_entry(snapshot_dir, dest, force=force)
            moved += 1
    return moved


def _migrate_retrieval_snapshots(
    snapshots_root: Path,
    dest_root: Path,
    *,
    force: bool,
    stats: Dict[str, int],
) -> int:
    moved = 0
    for manifest_path in snapshots_root.glob("*.json"):
        if not manifest_path.is_file():
            continue
        snapshot = RetrievalSnapshot.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        retriever_id = snapshot.configuration.retriever_id
        snapshot_dir = dest_root / retriever_id / snapshot.snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        updated_artifacts = []
        for artifact_path in snapshot.snapshot_artifacts:
            old_path = snapshots_root.parent / Path(artifact_path).relative_to(
                LEGACY_CORPUS_DIR_NAME
            )
            if not old_path.exists():
                old_path = snapshots_root / Path(artifact_path).name
            if not old_path.exists():
                if not force:
                    raise FileNotFoundError(f"Missing snapshot artifact: {artifact_path}")
                continue
            new_relpath = (
                Path(RETRIEVAL_DIR_NAME)
                / retriever_id
                / snapshot.snapshot_id
                / old_path.name
            )
            new_path = dest_root.parent / new_relpath
            new_path.parent.mkdir(parents=True, exist_ok=True)
            _move_entry(old_path, new_path, force=force)
            updated_artifacts.append(str(new_relpath))
            stats["updated_snapshot_artifacts"] += 1
        updated_snapshot = snapshot.model_copy(update={"snapshot_artifacts": updated_artifacts})
        manifest_out = snapshot_dir / "manifest.json"
        manifest_out.write_text(
            updated_snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        latest_path = dest_root / retriever_id / "latest.json"
        latest_path.write_text(
            json.dumps(
                {"snapshot_id": snapshot.snapshot_id, "created_at": snapshot.created_at},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path.unlink()
        moved += 1
    return moved


def _update_config_and_catalog(*, meta_dir: Path, stats: Dict[str, int]) -> None:
    config_path = meta_dir / "config.json"
    catalog_path = meta_dir / "catalog.json"
    if config_path.is_file():
        config = CorpusConfig.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
        updated_config = config.model_copy(update={"raw_dir": "."})
        config_path.write_text(updated_config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    else:
        default_config = CorpusConfig(
            schema_version=SCHEMA_VERSION,
            created_at=utc_now_iso(),
            corpus_uri=meta_dir.parent.resolve().as_uri(),
            raw_dir=".",
            notes=None,
            hooks=None,
            source=None,
            collection=None,
        )
        config_path.write_text(default_config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if catalog_path.is_file():
        catalog = CorpusCatalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))
        updated_items = {}
        for item_id, item in catalog.items.items():
            relpath = item.relpath
            if relpath.startswith("raw/"):
                relpath = relpath[len("raw/") :]
            updated_items[item_id] = item.model_copy(update={"relpath": relpath})
            stats["updated_catalog_items"] += 1
        updated_catalog = catalog.model_copy(
            update={
                "raw_dir": ".",
                "items": updated_items,
            }
        )
        catalog_path.write_text(
            updated_catalog.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


def _move_entry(source: Path, dest: Path, *, force: bool) -> None:
    if dest.exists():
        if not force:
            raise FileExistsError(f"Destination exists: {dest}")
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))


def _write_latest_pointers(root_dir: Path) -> None:
    if not root_dir.is_dir():
        return
    for extractor_dir in sorted(root_dir.iterdir()):
        if not extractor_dir.is_dir():
            continue
        latest_manifest = _select_latest_manifest(extractor_dir)
        if latest_manifest is None:
            continue
        latest_path = extractor_dir / "latest.json"
        latest_path.write_text(
            json.dumps(
                {
                    "snapshot_id": latest_manifest["snapshot_id"],
                    "created_at": latest_manifest["created_at"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _select_latest_manifest(extractor_dir: Path) -> Optional[Dict[str, str]]:
    latest: Optional[Dict[str, str]] = None
    for snapshot_dir in sorted(extractor_dir.iterdir()):
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        created_at = manifest.get("created_at")
        snapshot_id = manifest.get("snapshot_id")
        if not isinstance(created_at, str) or not isinstance(snapshot_id, str):
            continue
        if latest is None or created_at > latest["created_at"]:
            latest = {"snapshot_id": snapshot_id, "created_at": created_at}
    return latest
