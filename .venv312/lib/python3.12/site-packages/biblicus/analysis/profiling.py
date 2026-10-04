"""
Profiling analysis backend for Biblicus.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from pydantic import BaseModel

from ..corpus import Corpus
from ..models import CatalogItem, ExtractionSnapshotReference
from ..retrieval import hash_text
from ..time import utc_now_iso
from .base import CorpusAnalysisBackend
from .models import (
    AnalysisConfigurationManifest,
    AnalysisRunInput,
    AnalysisRunManifest,
    ProfilingConfiguration,
    ProfilingDistributionReport,
    ProfilingExtractedTextReport,
    ProfilingOutput,
    ProfilingPercentileValue,
    ProfilingRawItemsReport,
    ProfilingReport,
    ProfilingTagCount,
    ProfilingTagReport,
)


class ProfilingBackend(CorpusAnalysisBackend):
    """
    Profiling analysis backend for corpus composition and coverage.

    :ivar analysis_id: Backend identifier.
    :vartype analysis_id: str
    """

    analysis_id = "profiling"

    def run_analysis(
        self,
        corpus: Corpus,
        *,
        configuration_name: str,
        configuration: Dict[str, object],
        extraction_snapshot: ExtractionSnapshotReference,
    ) -> BaseModel:
        """
        Run the profiling analysis pipeline.

        :param corpus: Corpus to analyze.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Analysis configuration values.
        :type configuration: dict[str, object]
        :param extraction_snapshot: Extraction snapshot reference for text inputs.
        :type extraction_snapshot: biblicus.models.ExtractionSnapshotReference
        :return: Profiling output model.
        :rtype: pydantic.BaseModel
        """
        parsed_config = (
            configuration
            if isinstance(configuration, ProfilingConfiguration)
            else ProfilingConfiguration.model_validate(configuration)
        )
        return _run_profiling(
            corpus=corpus,
            configuration_name=configuration_name,
            config=parsed_config,
            extraction_snapshot=extraction_snapshot,
        )


def _run_profiling(
    *,
    corpus: Corpus,
    configuration_name: str,
    config: ProfilingConfiguration,
    extraction_snapshot: ExtractionSnapshotReference,
) -> ProfilingOutput:
    configuration_manifest = _create_configuration_manifest(
        name=configuration_name,
        config=config,
    )
    catalog = corpus.load_catalog()
    snapshot_id = _analysis_snapshot_id(
        configuration_id=configuration_manifest.configuration_id,
        extraction_snapshot=extraction_snapshot,
        catalog_generated_at=catalog.generated_at,
    )
    run_manifest = AnalysisRunManifest(
        snapshot_id=snapshot_id,
        configuration=configuration_manifest,
        corpus_uri=catalog.corpus_uri,
        catalog_generated_at=catalog.generated_at,
        created_at=utc_now_iso(),
        input=AnalysisRunInput(extraction_snapshot=extraction_snapshot),
        artifact_paths=[],
        stats={},
    )
    run_dir = corpus.analysis_run_dir(
        analysis_id=ProfilingBackend.analysis_id, snapshot_id=snapshot_id
    )
    output_path = run_dir / "output.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    ordered_items = _ordered_catalog_items(catalog.items, catalog.order)
    raw_report = _build_raw_items_report(items=ordered_items, config=config)
    extracted_report = _build_extracted_text_report(
        corpus=corpus,
        extraction_snapshot=extraction_snapshot,
        config=config,
    )

    report = ProfilingReport(
        raw_items=raw_report,
        extracted_text=extracted_report,
        warnings=[],
        errors=[],
    )

    run_stats = {
        "raw_items": raw_report.total_items,
        "extracted_nonempty_items": extracted_report.extracted_nonempty_items,
        "extracted_missing_items": extracted_report.extracted_missing_items,
    }
    run_manifest = run_manifest.model_copy(
        update={"artifact_paths": ["output.json"], "stats": run_stats}
    )
    _write_analysis_run_manifest(run_dir=run_dir, manifest=run_manifest)
    _write_latest_pointer(
        corpus=corpus,
        analysis_id=ProfilingBackend.analysis_id,
        manifest=run_manifest,
    )

    output = ProfilingOutput(
        analysis_id=ProfilingBackend.analysis_id,
        generated_at=utc_now_iso(),
        snapshot=run_manifest,
        report=report,
    )
    _write_profiling_output(path=output_path, output=output)
    return output


def _create_configuration_manifest(
    *, name: str, config: ProfilingConfiguration
) -> AnalysisConfigurationManifest:
    configuration_payload = json.dumps(
        {
            "analysis_id": ProfilingBackend.analysis_id,
            "name": name,
            "config": config.model_dump(),
        },
        sort_keys=True,
    )
    configuration_id = hash_text(configuration_payload)
    return AnalysisConfigurationManifest(
        configuration_id=configuration_id,
        analysis_id=ProfilingBackend.analysis_id,
        name=name,
        created_at=utc_now_iso(),
        config=config.model_dump(),
    )


def _analysis_snapshot_id(
    *,
    configuration_id: str,
    extraction_snapshot: ExtractionSnapshotReference,
    catalog_generated_at: str,
) -> str:
    run_seed = f"{configuration_id}:{extraction_snapshot.as_string()}:{catalog_generated_at}"
    return hash_text(run_seed)


def _ordered_catalog_items(
    items: Dict[str, CatalogItem],
    order: Sequence[str],
) -> List[CatalogItem]:
    ordered: List[CatalogItem] = []
    seen = set()
    for item_id in order:
        item = items.get(item_id)
        if item is None:
            continue
        ordered.append(item)
        seen.add(item_id)
    for item_id in sorted(items):
        if item_id in seen:
            continue
        ordered.append(items[item_id])
    return ordered


def _build_raw_items_report(
    *, items: Sequence[CatalogItem], config: ProfilingConfiguration
) -> ProfilingRawItemsReport:
    media_type_counts: Dict[str, int] = {}
    for item in items:
        media_type_counts[item.media_type] = media_type_counts.get(item.media_type, 0) + 1

    bytes_values = [item.bytes for item in _apply_sample(items, config.sample_size)]
    bytes_distribution = _build_distribution(bytes_values, config.percentiles)
    tag_report = _build_tag_report(items=items, config=config)

    return ProfilingRawItemsReport(
        total_items=len(items),
        media_type_counts=media_type_counts,
        bytes_distribution=bytes_distribution,
        tags=tag_report,
    )


def _build_tag_report(
    *, items: Sequence[CatalogItem], config: ProfilingConfiguration
) -> ProfilingTagReport:
    tag_filters = config.tag_filters
    tag_filter_set = set(tag_filters or [])
    tag_counts: Dict[str, int] = {}
    tagged_items = 0

    for item in items:
        tags = list(item.tags)
        if tag_filters is not None:
            tags = [tag for tag in tags if tag in tag_filter_set]
        if tags:
            tagged_items += 1
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    untagged_items = len(items) - tagged_items
    top_tags = sorted(tag_counts.items(), key=lambda entry: (-entry[1], entry[0]))
    top_tags = top_tags[: config.top_tag_count]
    return ProfilingTagReport(
        tagged_items=tagged_items,
        untagged_items=untagged_items,
        total_unique_tags=len(tag_counts),
        top_tags=[ProfilingTagCount(tag=tag, count=count) for tag, count in top_tags],
        tag_filters=tag_filters,
    )


def _build_extracted_text_report(
    *,
    corpus: Corpus,
    extraction_snapshot: ExtractionSnapshotReference,
    config: ProfilingConfiguration,
) -> ProfilingExtractedTextReport:
    manifest = corpus.load_extraction_snapshot_manifest(
        extractor_id=extraction_snapshot.extractor_id,
        snapshot_id=extraction_snapshot.snapshot_id,
    )
    nonempty_items = 0
    empty_items = 0
    missing_items = 0
    text_lengths: List[int] = []
    text_dir = corpus.extraction_snapshot_dir(
        extractor_id=extraction_snapshot.extractor_id,
        snapshot_id=extraction_snapshot.snapshot_id,
    )

    for item_result in manifest.items:
        if item_result.status != "extracted" or item_result.final_text_relpath is None:
            missing_items += 1
            continue
        text_path = text_dir / item_result.final_text_relpath
        text_value = text_path.read_text(encoding="utf-8")
        stripped = text_value.strip()
        if not stripped:
            empty_items += 1
            continue
        if config.min_text_characters is not None and len(stripped) < config.min_text_characters:
            empty_items += 1
            continue
        nonempty_items += 1
        text_lengths.append(len(stripped))

    sampled_lengths = _apply_sample(text_lengths, config.sample_size)
    characters_distribution = _build_distribution(sampled_lengths, config.percentiles)
    return ProfilingExtractedTextReport(
        source_items=len(manifest.items),
        extracted_nonempty_items=nonempty_items,
        extracted_empty_items=empty_items,
        extracted_missing_items=missing_items,
        characters_distribution=characters_distribution,
    )


def _apply_sample(values: Sequence, sample_size: int | None) -> List:
    if sample_size is None:
        return list(values)
    return list(values[:sample_size])


def _build_distribution(
    values: Sequence[int], percentiles: Iterable[int]
) -> ProfilingDistributionReport:
    if not values:
        percentile_values = [
            ProfilingPercentileValue(percentile=percentile, value=0.0) for percentile in percentiles
        ]
        return ProfilingDistributionReport(
            count=0,
            min_value=0.0,
            max_value=0.0,
            mean_value=0.0,
            percentiles=percentile_values,
        )
    sorted_values = sorted(values)
    count = len(sorted_values)
    min_value = float(sorted_values[0])
    max_value = float(sorted_values[-1])
    mean_value = float(sum(sorted_values)) / count
    percentile_values = [
        ProfilingPercentileValue(
            percentile=percentile,
            value=float(_percentile_value(sorted_values, percentile)),
        )
        for percentile in percentiles
    ]
    return ProfilingDistributionReport(
        count=count,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        percentiles=percentile_values,
    )


def _percentile_value(sorted_values: Sequence[int], percentile: int) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil((percentile / 100) * len(sorted_values)) - 1)
    index = min(index, len(sorted_values) - 1)
    return int(sorted_values[index])


def _write_analysis_run_manifest(*, run_dir: Path, manifest: AnalysisRunManifest) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_latest_pointer(
    *, corpus: Corpus, analysis_id: str, manifest: AnalysisRunManifest
) -> None:
    latest_path = corpus.analysis_dir / analysis_id / "latest.json"
    latest_path.write_text(
        json.dumps(
            {"snapshot_id": manifest.snapshot_id, "created_at": manifest.created_at},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_profiling_output(*, path: Path, output: ProfilingOutput) -> None:
    path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
