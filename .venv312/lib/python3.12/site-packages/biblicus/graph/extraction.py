"""
Graph extraction snapshots for Biblicus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..corpus import Corpus
from ..models import ExtractionSnapshotReference
from ..retrieval import hash_text
from ..time import utc_now_iso
from .extractors import get_graph_extractor
from .models import (
    GraphConfigurationManifest,
    GraphExtractionItemSummary,
    GraphExtractionResult,
    GraphSnapshotListEntry,
    GraphSnapshotManifest,
    GraphSnapshotReference,
    parse_graph_snapshot_reference,
)
from .neo4j import create_neo4j_driver, resolve_neo4j_settings, write_graph_records


def create_graph_configuration_manifest(
    *, extractor_id: str, name: str, configuration: Dict[str, Any]
) -> GraphConfigurationManifest:
    """
    Create a deterministic graph extraction configuration manifest.

    :param extractor_id: Graph extractor identifier.
    :type extractor_id: str
    :param name: Human configuration name.
    :type name: str
    :param configuration: Extractor configuration.
    :type configuration: dict[str, Any]
    :return: Configuration manifest.
    :rtype: GraphConfigurationManifest
    """
    configuration_payload = json.dumps(
        {"extractor_id": extractor_id, "name": name, "configuration": configuration},
        sort_keys=True,
    )
    configuration_id = hash_text(configuration_payload)
    return GraphConfigurationManifest(
        configuration_id=configuration_id,
        extractor_id=extractor_id,
        name=name,
        created_at=utc_now_iso(),
        configuration=configuration,
    )


def create_graph_id(*, extractor_id: str, configuration: Dict[str, Any]) -> str:
    """
    Create a deterministic graph identifier from extractor and configuration.

    :param extractor_id: Graph extractor identifier.
    :type extractor_id: str
    :param configuration: Extractor configuration.
    :type configuration: dict[str, Any]
    :return: Graph identifier.
    :rtype: str
    """
    config_payload = json.dumps(configuration, sort_keys=True)
    config_hash = hash_text(config_payload)
    return f"{extractor_id}:{config_hash}"


def create_graph_snapshot_manifest(
    corpus: Corpus,
    *,
    configuration: GraphConfigurationManifest,
    extraction_snapshot: ExtractionSnapshotReference,
    graph_id: str,
) -> GraphSnapshotManifest:
    """
    Create a new graph snapshot manifest for a corpus.

    :param corpus: Corpus associated with the snapshot.
    :type corpus: Corpus
    :param configuration: Configuration manifest.
    :type configuration: GraphConfigurationManifest
    :param extraction_snapshot: Extraction snapshot reference.
    :type extraction_snapshot: ExtractionSnapshotReference
    :param graph_id: Graph identifier.
    :type graph_id: str
    :return: Graph snapshot manifest.
    :rtype: GraphSnapshotManifest
    """
    catalog = corpus.load_catalog()
    snapshot_id = hash_text(
        f"{configuration.configuration_id}:{extraction_snapshot.as_string()}:{catalog.generated_at}"
    )
    return GraphSnapshotManifest(
        snapshot_id=snapshot_id,
        graph_id=graph_id,
        configuration=configuration,
        corpus_uri=corpus.uri,
        catalog_generated_at=catalog.generated_at,
        extraction_snapshot=extraction_snapshot.as_string(),
        created_at=utc_now_iso(),
        stats={},
    )


def write_graph_snapshot_manifest(*, snapshot_dir: Path, manifest: GraphSnapshotManifest) -> None:
    """
    Persist a graph snapshot manifest to a snapshot directory.

    :param snapshot_dir: Graph snapshot directory.
    :type snapshot_dir: Path
    :param manifest: Snapshot manifest to write.
    :type manifest: GraphSnapshotManifest
    :return: None.
    :rtype: None
    """
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_graph_latest_pointer(*, extractor_dir: Path, manifest: GraphSnapshotManifest) -> None:
    """
    Persist the latest pointer for a graph extractor.

    :param extractor_dir: Extractor directory containing snapshots.
    :type extractor_dir: Path
    :param manifest: Snapshot manifest used for the pointer.
    :type manifest: GraphSnapshotManifest
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


def build_graph_snapshot(
    corpus: Corpus,
    *,
    extractor_id: str,
    configuration_name: str,
    configuration: Dict[str, Any],
    extraction_snapshot: ExtractionSnapshotReference,
) -> GraphSnapshotManifest:
    """
    Build a graph extraction snapshot for a corpus.

    :param corpus: Corpus to process.
    :type corpus: Corpus
    :param extractor_id: Graph extractor identifier.
    :type extractor_id: str
    :param configuration_name: Human configuration name.
    :type configuration_name: str
    :param configuration: Extractor configuration values.
    :type configuration: dict[str, Any]
    :param extraction_snapshot: Extraction snapshot reference.
    :type extraction_snapshot: ExtractionSnapshotReference
    :return: Graph snapshot manifest.
    :rtype: GraphSnapshotManifest
    """
    extractor = get_graph_extractor(extractor_id)
    try:
        parsed_config = extractor.validate_config(configuration)
    except ValidationError as exc:
        raise ValueError(f"Invalid graph extraction configuration: {exc}") from exc

    graph_id = create_graph_id(extractor_id=extractor_id, configuration=configuration)
    configuration_manifest = create_graph_configuration_manifest(
        extractor_id=extractor_id,
        name=configuration_name,
        configuration=configuration,
    )
    manifest = create_graph_snapshot_manifest(
        corpus,
        configuration=configuration_manifest,
        extraction_snapshot=extraction_snapshot,
        graph_id=graph_id,
    )
    extraction_manifest = corpus.load_extraction_snapshot_manifest(
        extractor_id=extraction_snapshot.extractor_id,
        snapshot_id=extraction_snapshot.snapshot_id,
    )

    snapshot_dir = corpus.graph_snapshot_dir(
        extractor_id=extractor_id,
        snapshot_id=manifest.snapshot_id,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    settings = resolve_neo4j_settings()
    driver = create_neo4j_driver(settings)

    node_total = 0
    edge_total = 0
    item_summaries: List[GraphExtractionItemSummary] = []

    try:
        for item_result in extraction_manifest.items:
            item = corpus.get_item(item_result.item_id)
            extracted_text = _load_extracted_text(
                corpus,
                extraction_snapshot=extraction_snapshot,
                item_result=item_result,
            )
            if extracted_text is None:
                item_summaries.append(
                    GraphExtractionItemSummary(
                        item_id=item.id,
                        status="skipped",
                        node_count=0,
                        edge_count=0,
                        error_message="No extracted text",
                    )
                )
                continue
            result = extractor.extract_graph(
                corpus=corpus,
                item=item,
                extracted_text=extracted_text,
                config=parsed_config,
            )
            if not isinstance(result, GraphExtractionResult):
                raise ValueError("Graph extractor must return GraphExtractionResult")
            write_graph_records(
                driver=driver,
                settings=settings,
                corpus_id=corpus.uri,
                graph_id=graph_id,
                extraction_snapshot=extraction_snapshot.as_string(),
                item_id=item.id,
                nodes=result.nodes,
                edges=result.edges,
            )
            node_total += len(result.nodes)
            edge_total += len(result.edges)
            item_summaries.append(
                GraphExtractionItemSummary(
                    item_id=item.id,
                    status="complete",
                    node_count=len(result.nodes),
                    edge_count=len(result.edges),
                )
            )
    finally:
        driver.close()

    manifest.stats = {
        "items_total": len(extraction_manifest.items),
        "items_processed": len(item_summaries),
        "nodes": node_total,
        "edges": edge_total,
    }
    write_graph_snapshot_manifest(snapshot_dir=snapshot_dir, manifest=manifest)
    write_graph_latest_pointer(extractor_dir=snapshot_dir.parent, manifest=manifest)
    return manifest


def _load_extracted_text(
    corpus: Corpus,
    *,
    extraction_snapshot: ExtractionSnapshotReference,
    item_result,
) -> Optional[str]:
    if not item_result.final_text_relpath:
        return None
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=extraction_snapshot.extractor_id,
        snapshot_id=extraction_snapshot.snapshot_id,
    )
    text_path = snapshot_dir / item_result.final_text_relpath
    if not text_path.is_file():
        return None
    return text_path.read_text(encoding="utf-8")


def load_graph_snapshot_manifest(
    corpus: Corpus, *, extractor_id: str, snapshot_id: str
) -> GraphSnapshotManifest:
    """
    Load a graph snapshot manifest from the corpus.

    :param corpus: Corpus containing the snapshot.
    :type corpus: Corpus
    :param extractor_id: Graph extractor identifier.
    :type extractor_id: str
    :param snapshot_id: Graph snapshot identifier.
    :type snapshot_id: str
    :return: Parsed snapshot manifest.
    :rtype: GraphSnapshotManifest
    :raises FileNotFoundError: If the manifest file does not exist.
    :raises ValueError: If the manifest data is invalid.
    """
    manifest_path = (
        corpus.graph_snapshot_dir(extractor_id=extractor_id, snapshot_id=snapshot_id)
        / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing graph snapshot manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return GraphSnapshotManifest.model_validate(data)


def list_graph_snapshots(
    corpus: Corpus, *, extractor_id: Optional[str] = None
) -> List[GraphSnapshotListEntry]:
    """
    List graph snapshots stored under the corpus.

    :param corpus: Corpus containing the snapshots.
    :type corpus: Corpus
    :param extractor_id: Optional extractor identifier filter.
    :type extractor_id: str or None
    :return: Summary list entries for each snapshot.
    :rtype: list[GraphSnapshotListEntry]
    """
    snapshots_root = corpus.graph_snapshots_dir
    if not snapshots_root.is_dir():
        return []

    extractor_dirs: List[Path]
    if extractor_id is None:
        extractor_dirs = [path for path in sorted(snapshots_root.iterdir()) if path.is_dir()]
    else:
        extractor_path = snapshots_root / extractor_id
        extractor_dirs = [extractor_path] if extractor_path.is_dir() else []

    entries: List[GraphSnapshotListEntry] = []
    for extractor_dir in extractor_dirs:
        for snapshot_dir in sorted(extractor_dir.iterdir()):
            if not snapshot_dir.is_dir():
                continue
            manifest_path = snapshot_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = load_graph_snapshot_manifest(
                    corpus,
                    extractor_id=extractor_dir.name,
                    snapshot_id=snapshot_dir.name,
                )
            except (FileNotFoundError, ValueError):
                continue
            entries.append(
                GraphSnapshotListEntry(
                    extractor_id=extractor_dir.name,
                    snapshot_id=snapshot_dir.name,
                    graph_id=manifest.graph_id,
                    configuration_id=manifest.configuration.configuration_id,
                    configuration_name=manifest.configuration.name,
                    catalog_generated_at=manifest.catalog_generated_at,
                    created_at=manifest.created_at,
                    stats=dict(manifest.stats),
                )
            )

    entries.sort(
        key=lambda entry: (entry.created_at, entry.extractor_id, entry.snapshot_id),
        reverse=True,
    )
    return entries


def latest_graph_snapshot_reference(
    corpus: Corpus, *, extractor_id: Optional[str] = None
) -> Optional[GraphSnapshotReference]:
    """
    Return the most recent graph snapshot reference.

    :param corpus: Corpus containing the snapshots.
    :type corpus: Corpus
    :param extractor_id: Optional extractor identifier filter.
    :type extractor_id: str or None
    :return: Latest graph snapshot reference or None when no snapshots exist.
    :rtype: GraphSnapshotReference or None
    """
    entries = list_graph_snapshots(corpus, extractor_id=extractor_id)
    if not entries:
        return None
    latest = entries[0]
    return GraphSnapshotReference(extractor_id=latest.extractor_id, snapshot_id=latest.snapshot_id)


def resolve_graph_snapshot_reference(
    corpus: Corpus, *, raw: str
) -> GraphSnapshotReference:
    """
    Resolve a graph snapshot reference from a raw string.

    :param corpus: Corpus containing the snapshots.
    :type corpus: Corpus
    :param raw: Raw snapshot reference.
    :type raw: str
    :return: Parsed graph snapshot reference.
    :rtype: GraphSnapshotReference
    """
    return parse_graph_snapshot_reference(raw)
