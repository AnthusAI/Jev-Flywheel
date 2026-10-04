"""
Topic modeling analysis backend for Biblicus.
"""

from __future__ import annotations

import json
import re
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..ai.llm import generate_completion
from ..corpus import Corpus
from ..models import ExtractionSnapshotReference
from ..retrieval import hash_text
from ..time import utc_now_iso
from .base import CorpusAnalysisBackend
from .models import (
    AnalysisConfigurationManifest,
    AnalysisRunInput,
    AnalysisRunManifest,
    TopicModelingBerTopicConfig,
    TopicModelingBerTopicReport,
    TopicModelingConfiguration,
    TopicModelingEntityRemovalConfig,
    TopicModelingEntityRemovalReport,
    TopicModelingKeyword,
    TopicModelingLabelSource,
    TopicModelingLexicalProcessingConfig,
    TopicModelingLexicalProcessingReport,
    TopicModelingLlmExtractionConfig,
    TopicModelingLlmExtractionMethod,
    TopicModelingLlmExtractionReport,
    TopicModelingLlmFineTuningConfig,
    TopicModelingLlmFineTuningReport,
    TopicModelingOutput,
    TopicModelingReport,
    TopicModelingStageStatus,
    TopicModelingTextCollectionReport,
    TopicModelingTextSourceConfig,
    TopicModelingTopic,
)

_DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "GPE",
    "LOC",
    "ORG",
    "FAC",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "CARDINAL",
    "ORDINAL",
]


@dataclass
class TopicModelingDocument:
    """
    Text document input for topic modeling.

    :ivar document_id: Stable identifier for this document in the topic modeling stage.
    :vartype document_id: str
    :ivar source_item_id: Corpus item identifier the text was derived from.
    :vartype source_item_id: str
    :ivar text: Document text content.
    :vartype text: str
    """

    document_id: str
    source_item_id: str
    text: str


class TopicModelingBackend(CorpusAnalysisBackend):
    """
    Topic modeling analysis backend backed by BERTopic.

    :ivar analysis_id: Backend identifier.
    :vartype analysis_id: str
    """

    analysis_id = "topic-modeling"

    def run_analysis(
        self,
        corpus: Corpus,
        *,
        configuration_name: str,
        configuration: Dict[str, object],
        extraction_snapshot: ExtractionSnapshotReference,
    ) -> BaseModel:
        """
        Run the topic modeling analysis pipeline.

        :param corpus: Corpus to analyze.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Analysis configuration values.
        :type configuration: dict[str, object]
        :param extraction_snapshot: Extraction snapshot reference for text inputs.
        :type extraction_snapshot: biblicus.models.ExtractionSnapshotReference
        :return: Topic modeling output model.
        :rtype: pydantic.BaseModel
        """
        parsed_config = (
            configuration
            if isinstance(configuration, TopicModelingConfiguration)
            else TopicModelingConfiguration.model_validate(configuration)
        )
        return _run_topic_modeling(
            corpus=corpus,
            configuration_name=configuration_name,
            config=parsed_config,
            extraction_snapshot=extraction_snapshot,
        )


def _run_topic_modeling(
    *,
    corpus: Corpus,
    configuration_name: str,
    config: TopicModelingConfiguration,
    extraction_snapshot: ExtractionSnapshotReference,
) -> TopicModelingOutput:
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
        analysis_id=TopicModelingBackend.analysis_id, snapshot_id=snapshot_id
    )
    output_path = run_dir / "output.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    documents, text_report = _collect_documents(
        corpus=corpus,
        extraction_snapshot=extraction_snapshot,
        config=config.text_source,
    )

    llm_extraction_cache_path = _llm_extraction_cache_path(corpus=corpus, config=config)
    if llm_extraction_cache_path is not None:
        _seed_llm_extraction_cache(
            analysis_root=corpus.analysis_dir / TopicModelingBackend.analysis_id,
            cache_path=llm_extraction_cache_path,
            config=config.llm_extraction,
            documents=documents,
        )
    llm_extraction_report, extracted_documents = _apply_llm_extraction(
        documents=documents,
        config=config.llm_extraction,
        cache_path=llm_extraction_cache_path,
    )
    llm_extraction_path = run_dir / "llm_extraction.jsonl"
    if config.llm_extraction.enabled:
        _write_documents_jsonl(llm_extraction_path, extracted_documents)

    entity_removal_path = run_dir / "entity_removal.jsonl"
    entity_removal_report, entity_documents = _apply_entity_removal(
        documents=extracted_documents,
        config=config.entity_removal,
        cache_path=entity_removal_path,
    )

    lexical_report, lexical_documents = _apply_lexical_processing(
        documents=entity_documents,
        config=config.lexical_processing,
    )

    bertopic_report, topics = _run_bertopic(
        documents=lexical_documents,
        config=config.bertopic_analysis,
    )

    fine_tuning_report, labeled_topics = _apply_llm_fine_tuning(
        topics=topics,
        documents=lexical_documents,
        config=config.llm_fine_tuning,
    )

    report = TopicModelingReport(
        text_collection=text_report,
        llm_extraction=llm_extraction_report,
        entity_removal=entity_removal_report,
        lexical_processing=lexical_report,
        bertopic_analysis=bertopic_report,
        llm_fine_tuning=fine_tuning_report,
        topics=labeled_topics,
        warnings=(
            text_report.warnings
            + llm_extraction_report.warnings
            + entity_removal_report.warnings
            + bertopic_report.warnings
            + fine_tuning_report.warnings
        ),
        errors=text_report.errors
        + llm_extraction_report.errors
        + entity_removal_report.errors
        + bertopic_report.errors
        + fine_tuning_report.errors,
    )

    run_stats = {
        "documents": bertopic_report.document_count,
        "topics": bertopic_report.topic_count,
    }
    artifact_paths = ["output.json"]
    if config.llm_extraction.enabled:
        artifact_paths.append("llm_extraction.jsonl")
    if config.entity_removal.enabled:
        artifact_paths.append("entity_removal.jsonl")

    run_manifest = run_manifest.model_copy(
        update={"artifact_paths": artifact_paths, "stats": run_stats}
    )
    _write_analysis_run_manifest(run_dir=run_dir, manifest=run_manifest)
    _write_latest_pointer(
        corpus=corpus,
        analysis_id=TopicModelingBackend.analysis_id,
        manifest=run_manifest,
    )

    output = TopicModelingOutput(
        analysis_id=TopicModelingBackend.analysis_id,
        generated_at=utc_now_iso(),
        snapshot=run_manifest,
        report=report,
    )
    _write_topic_modeling_output(path=output_path, output=output)
    return output


def run_topic_modeling_for_documents(
    *,
    documents: List[TopicModelingDocument],
    config: TopicModelingConfiguration,
    artifacts_dir: Optional[Path] = None,
) -> TopicModelingReport:
    """
    Run topic modeling using caller-provided documents.

    :param documents: Pre-collected documents to model.
    :type documents: list[TopicModelingDocument]
    :param config: Topic modeling configuration.
    :type config: TopicModelingConfiguration
    :return: Topic modeling report with topic assignments.
    :rtype: TopicModelingReport
    """
    text_report = TopicModelingTextCollectionReport(
        status=TopicModelingStageStatus.COMPLETE,
        source_items=len({doc.source_item_id for doc in documents}),
        documents=len(documents),
        sample_size=config.text_source.sample_size,
        min_text_characters=config.text_source.min_text_characters,
        empty_texts=len([doc for doc in documents if not doc.text.strip()]),
        skipped_items=0,
        warnings=[],
        errors=[],
    )

    llm_extraction_report, extracted_documents = _apply_llm_extraction(
        documents=documents,
        config=config.llm_extraction,
    )
    if artifacts_dir is not None and config.llm_extraction.enabled:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        _write_documents_jsonl(artifacts_dir / "llm_extraction.jsonl", extracted_documents)

    entity_removal_path = (
        artifacts_dir / "entity_removal.jsonl" if artifacts_dir is not None else None
    )
    entity_removal_report, entity_documents = _apply_entity_removal(
        documents=extracted_documents,
        config=config.entity_removal,
        cache_path=entity_removal_path,
    )

    lexical_report, lexical_documents = _apply_lexical_processing(
        documents=entity_documents,
        config=config.lexical_processing,
    )

    bertopic_report, topics = _run_bertopic(
        documents=lexical_documents,
        config=config.bertopic_analysis,
    )

    fine_tuning_report, labeled_topics = _apply_llm_fine_tuning(
        topics=topics,
        documents=lexical_documents,
        config=config.llm_fine_tuning,
    )

    return TopicModelingReport(
        text_collection=text_report,
        llm_extraction=llm_extraction_report,
        entity_removal=entity_removal_report,
        lexical_processing=lexical_report,
        bertopic_analysis=bertopic_report,
        llm_fine_tuning=fine_tuning_report,
        topics=labeled_topics,
        warnings=(
            text_report.warnings
            + llm_extraction_report.warnings
            + entity_removal_report.warnings
            + bertopic_report.warnings
            + fine_tuning_report.warnings
        ),
        errors=text_report.errors
        + llm_extraction_report.errors
        + entity_removal_report.errors
        + bertopic_report.errors
        + fine_tuning_report.errors,
    )


def _create_configuration_manifest(
    *, name: str, config: TopicModelingConfiguration
) -> AnalysisConfigurationManifest:
    configuration_payload = json.dumps(
        {
            "analysis_id": TopicModelingBackend.analysis_id,
            "name": name,
            "config": config.model_dump(),
        },
        sort_keys=True,
    )
    configuration_id = hash_text(configuration_payload)
    return AnalysisConfigurationManifest(
        configuration_id=configuration_id,
        analysis_id=TopicModelingBackend.analysis_id,
        name=name,
        created_at=utc_now_iso(),
        config=config.model_dump(),
    )


def _llm_extraction_cache_identity(config: TopicModelingLlmExtractionConfig) -> str:
    payload = config.model_dump()
    payload.pop("max_workers", None)
    client_payload = payload.get("client")
    if isinstance(client_payload, dict):
        client_payload.pop("api_key", None)
    return hash_text(json.dumps(payload, sort_keys=True))


def _llm_extraction_cache_path(
    *, corpus: Corpus, config: TopicModelingConfiguration
) -> Optional[Path]:
    if not config.llm_extraction.enabled:
        return None
    cache_id = _llm_extraction_cache_identity(config.llm_extraction)
    return (
        corpus.analysis_dir
        / TopicModelingBackend.analysis_id
        / "cache"
        / "llm-extraction"
        / f"{cache_id}.jsonl"
    )


def _analysis_snapshot_id(
    *,
    configuration_id: str,
    extraction_snapshot: ExtractionSnapshotReference,
    catalog_generated_at: str,
) -> str:
    run_seed = f"{configuration_id}:{extraction_snapshot.as_string()}:{catalog_generated_at}"
    return hash_text(run_seed)


def _collect_documents(
    *,
    corpus: Corpus,
    extraction_snapshot: ExtractionSnapshotReference,
    config: TopicModelingTextSourceConfig,
) -> Tuple[List[TopicModelingDocument], TopicModelingTextCollectionReport]:
    manifest = corpus.load_extraction_snapshot_manifest(
        extractor_id=extraction_snapshot.extractor_id,
        snapshot_id=extraction_snapshot.snapshot_id,
    )
    warnings: List[str] = []
    errors: List[str] = []
    documents: List[TopicModelingDocument] = []
    skipped_items = 0
    empty_texts = 0

    for item_result in manifest.items:
        if item_result.status != "extracted" or item_result.final_text_relpath is None:
            skipped_items += 1
            continue
        text_path = (
            corpus.extraction_snapshot_dir(
                extractor_id=extraction_snapshot.extractor_id,
                snapshot_id=extraction_snapshot.snapshot_id,
            )
            / item_result.final_text_relpath
        )
        text_value = text_path.read_text(encoding="utf-8").strip()
        if not text_value:
            empty_texts += 1
            continue
        if config.min_text_characters is not None and len(text_value) < config.min_text_characters:
            skipped_items += 1
            continue
        documents.append(
            TopicModelingDocument(
                document_id=item_result.item_id,
                source_item_id=item_result.item_id,
                text=text_value,
            )
        )

    if config.sample_size is not None and len(documents) > config.sample_size:
        documents = documents[: config.sample_size]
        warnings.append("Text collection truncated to sample_size")

    report = TopicModelingTextCollectionReport(
        status=TopicModelingStageStatus.COMPLETE,
        source_items=len(manifest.items),
        documents=len(documents),
        sample_size=config.sample_size,
        min_text_characters=config.min_text_characters,
        empty_texts=empty_texts,
        skipped_items=skipped_items,
        warnings=warnings,
        errors=errors,
    )
    if not documents:
        raise ValueError("Topic modeling requires at least one extracted text document")
    return documents, report


def _apply_llm_extraction(
    *,
    documents: List[TopicModelingDocument],
    config: TopicModelingLlmExtractionConfig,
    cache_path: Optional[Path] = None,
) -> Tuple[TopicModelingLlmExtractionReport, List[TopicModelingDocument]]:
    if not config.enabled:
        report = TopicModelingLlmExtractionReport(
            status=TopicModelingStageStatus.SKIPPED,
            method=config.method,
            input_documents=len(documents),
            output_documents=len(documents),
            warnings=[],
            errors=[],
        )
        return report, list(documents)

    extracted_documents: List[TopicModelingDocument] = []
    errors: List[str] = []
    fatal_errors: List[str] = []
    cached_outputs = _read_llm_extraction_cache(cache_path)
    generated_outputs: Dict[str, List[TopicModelingDocument]] = {}
    cache_lock = threading.Lock()
    total = len(documents)
    if total <= 50:
        log_interval = 10
    elif total <= 200:
        log_interval = 25
    elif total <= 1000:
        log_interval = 50
    else:
        log_interval = 100
    start_time = time.perf_counter()
    last_log_time = start_time
    completed = 0

    print(
        f"[topic-modeling] llm extraction starting documents={total} "
        f"workers={config.max_workers}",
        flush=True,
        file=sys.stderr,
    )

    def log_progress() -> None:
        nonlocal last_log_time
        now = time.perf_counter()
        if completed % log_interval == 0 or completed == total or now - last_log_time >= 30.0:
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[topic-modeling] llm extraction {completed}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
                file=sys.stderr,
            )
            last_log_time = now

    def extract_document(
        document: TopicModelingDocument,
    ) -> Tuple[List[TopicModelingDocument], List[str]]:
        prompt = config.prompt_template.format(text=document.text)
        response_text = generate_completion(
            client=config.client,
            system_prompt=config.system_prompt,
            user_prompt=prompt,
        ).strip()
        if config.method == TopicModelingLlmExtractionMethod.SINGLE:
            if not response_text:
                return [], [
                    f"LLM extraction returned empty output for {document.document_id}"
                ]
            return [
                TopicModelingDocument(
                    document_id=document.document_id,
                    source_item_id=document.source_item_id,
                    text=response_text,
                )
            ], []
        items = _parse_itemized_response(response_text)
        if not items:
            return [], [f"LLM itemization returned no items for {document.document_id}"]
        return [
            TopicModelingDocument(
                document_id=f"{document.document_id}:{index}",
                source_item_id=document.source_item_id,
                text=item_text,
            )
            for index, item_text in enumerate(items, start=1)
        ], []

    def record_generated(
        document: TopicModelingDocument, output_documents: List[TopicModelingDocument]
    ) -> None:
        if cache_path is None or not output_documents:
            return
        cache_key = _llm_extraction_document_cache_key(document)
        with cache_lock:
            if cache_key in cached_outputs or cache_key in generated_outputs:
                return
            generated_outputs[cache_key] = list(output_documents)
            _append_llm_extraction_cache_entry(
                path=cache_path,
                cache_key=cache_key,
                documents=output_documents,
            )

    ordered_outputs: List[List[TopicModelingDocument]] = [[] for _ in documents]
    to_generate: List[tuple[int, TopicModelingDocument]] = []
    for index, document in enumerate(documents):
        cached = cached_outputs.get(_llm_extraction_document_cache_key(document))
        if cached is not None:
            ordered_outputs[index] = cached
            completed += 1
            log_progress()
        else:
            to_generate.append((index, document))

    def record_result(index: int, document: TopicModelingDocument) -> None:
        nonlocal completed
        try:
            output_documents, result_errors = extract_document(document)
        except Exception as exc:  # noqa: BLE001
            output_documents = []
            result_errors = []
            fatal_errors.append(f"LLM extraction failed for {document.document_id}: {exc}")
        ordered_outputs[index] = output_documents
        if result_errors:
            errors.extend(result_errors)
        else:
            record_generated(document, output_documents)
        completed += 1
        log_progress()

    if config.max_workers == 1 or len(to_generate) <= 1:
        for index, document in to_generate:
            record_result(index, document)
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(extract_document, document): (index, document)
                for index, document in to_generate
            }
            for future in as_completed(futures):
                index, document = futures[future]
                try:
                    output_documents, result_errors = future.result()
                except Exception as exc:  # noqa: BLE001
                    output_documents = []
                    result_errors = []
                    fatal_errors.append(
                        f"LLM extraction failed for {document.document_id}: {exc}"
                    )
                ordered_outputs[index] = output_documents
                if result_errors:
                    errors.extend(result_errors)
                else:
                    record_generated(document, output_documents)
                completed += 1
                log_progress()

    for output_documents in ordered_outputs:
        for output_document in output_documents:
            extracted_documents.append(output_document)

    report_warnings: List[str] = []
    cached_count = total - len(to_generate)
    if cached_count:
        report_warnings.append(f"Reused cached LLM extraction documents for {cached_count} inputs")

    report = TopicModelingLlmExtractionReport(
        status=TopicModelingStageStatus.COMPLETE,
        method=config.method,
        input_documents=len(documents),
        output_documents=len(extracted_documents),
        warnings=report_warnings,
        errors=fatal_errors + errors,
    )
    if not extracted_documents:
        if fatal_errors:
            raise ValueError("; ".join(fatal_errors))
        raise ValueError("LLM extraction produced no usable documents")
    return report, extracted_documents


def _llm_extraction_document_cache_key(document: TopicModelingDocument) -> str:
    return hash_text(f"{document.document_id}:{document.source_item_id}:{document.text}")


def _append_llm_extraction_cache_entry(
    *, path: Path, cache_key: str, documents: List[TopicModelingDocument]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "documents": [
            {
                "document_id": document.document_id,
                "source_item_id": document.source_item_id,
                "text": document.text,
            }
            for document in documents
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _read_llm_extraction_cache(
    path: Optional[Path],
) -> Dict[str, List[TopicModelingDocument]]:
    if path is None or not path.exists():
        return {}
    cached: Dict[str, List[TopicModelingDocument]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            cache_key = str(payload.get("cache_key") or "")
            documents_payload = payload.get("documents")
            if not cache_key or not isinstance(documents_payload, list):
                continue
            cached[cache_key] = [
                TopicModelingDocument(
                    document_id=str(document_payload.get("document_id", "")),
                    source_item_id=str(document_payload.get("source_item_id", "")),
                    text=str(document_payload.get("text", "")),
                )
                for document_payload in documents_payload
                if isinstance(document_payload, dict)
            ]
    return cached


def _seed_llm_extraction_cache(
    *,
    analysis_root: Path,
    cache_path: Path,
    config: TopicModelingLlmExtractionConfig,
    documents: List[TopicModelingDocument],
) -> None:
    existing_cache = _read_llm_extraction_cache(cache_path)
    source_documents = {document.document_id: document for document in documents}
    target_identity = _llm_extraction_cache_identity(config)
    seeded = 0
    for output_path in analysis_root.glob("*/output.json"):
        llm_path = output_path.parent / "llm_extraction.jsonl"
        if not llm_path.exists():
            continue
        try:
            output_payload = json.loads(output_path.read_text(encoding="utf-8"))
            output_config = TopicModelingConfiguration.model_validate(
                output_payload["snapshot"]["configuration"]["config"]
            )
        except (KeyError, OSError, json.JSONDecodeError, ValueError):
            continue
        if _llm_extraction_cache_identity(output_config.llm_extraction) != target_identity:
            continue
        grouped: Dict[str, List[TopicModelingDocument]] = {}
        for output_document in _read_documents_jsonl(llm_path):
            source_document_id = output_document.document_id.split(":", 1)[0]
            if source_document_id not in source_documents:
                continue
            grouped.setdefault(source_document_id, []).append(output_document)
        for source_document_id, output_documents in grouped.items():
            cache_key = _llm_extraction_document_cache_key(source_documents[source_document_id])
            if cache_key in existing_cache:
                continue
            _append_llm_extraction_cache_entry(
                path=cache_path,
                cache_key=cache_key,
                documents=output_documents,
            )
            existing_cache[cache_key] = output_documents
            seeded += 1
    if seeded:
        print(
            f"[topic-modeling] seeded {seeded} LLM extraction cache entries",
            flush=True,
            file=sys.stderr,
        )


def _remove_entities_from_text(
    *, text: str, entities: List[object], entity_types: set[str], replace_with: str
) -> str:
    spans: List[tuple[int, int]] = []
    for entity in entities:
        label = str(getattr(entity, "label_", "")).upper()
        if label and label not in entity_types:
            continue
        start = int(getattr(entity, "start_char", 0))
        end = int(getattr(entity, "end_char", 0))
        if end <= start:
            continue
        spans.append((start, end))
    if not spans:
        return text
    spans.sort()
    cleaned_parts: List[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        cleaned_parts.append(text[cursor:start])
        if replace_with:
            cleaned_parts.append(replace_with)
        cursor = end
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts)


def _apply_entity_removal(
    *,
    documents: List[TopicModelingDocument],
    config: TopicModelingEntityRemovalConfig,
    cache_path: Optional[Path] = None,
) -> Tuple[TopicModelingEntityRemovalReport, List[TopicModelingDocument]]:
    if not config.enabled:
        report = TopicModelingEntityRemovalReport(
            status=TopicModelingStageStatus.SKIPPED,
            provider=str(config.provider),
            model=str(config.model),
            entity_types=[],
            input_documents=len(documents),
            output_documents=len(documents),
            regex_patterns=[],
            warnings=[],
            errors=[],
        )
        return report, list(documents)

    if cache_path is not None and cache_path.exists():
        cached_documents = _read_documents_jsonl(cache_path)
        report = TopicModelingEntityRemovalReport(
            status=TopicModelingStageStatus.COMPLETE,
            provider=str(config.provider),
            model=str(config.model),
            entity_types=sorted(
                {entry.strip().upper() for entry in config.entity_types if str(entry).strip()}
                if config.entity_types
                else set(_DEFAULT_ENTITY_TYPES)
            ),
            input_documents=len(documents),
            output_documents=len(cached_documents),
            regex_patterns=list(config.regex_patterns),
            warnings=["Reused cached entity removal documents"],
            errors=[],
        )
        return report, cached_documents

    if config.provider.strip().lower() != "spacy":
        raise ValueError("entity_removal.provider must be 'spacy'")

    try:
        import spacy
    except ImportError as import_error:
        raise ValueError(
            'Entity removal requires spaCy. Install it with pip install "biblicus[ner]".'
        ) from import_error

    try:
        nlp = spacy.load(config.model)
    except Exception as load_error:
        raise ValueError(
            f"Entity removal requires spaCy model '{config.model}'. "
            "Install it with: python -m spacy download "
            f"{config.model}"
        ) from load_error

    entity_types = (
        {entry.strip().upper() for entry in config.entity_types if str(entry).strip()}
        if config.entity_types
        else set(_DEFAULT_ENTITY_TYPES)
    )
    processed: List[TopicModelingDocument] = []
    total = len(documents)
    if total <= 50:
        log_interval = 10
    elif total <= 200:
        log_interval = 25
    elif total <= 1000:
        log_interval = 50
    else:
        log_interval = 100
    start_time = time.perf_counter()
    last_log_time = start_time
    completed = 0

    def log_progress() -> None:
        nonlocal last_log_time
        now = time.perf_counter()
        if completed % log_interval == 0 or completed == total or now - last_log_time >= 30.0:
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[topic-modeling] entity removal {completed}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
                file=sys.stderr,
            )
            last_log_time = now

    for document in documents:
        text_value = document.text
        doc_obj = nlp(text_value)
        text_value = _remove_entities_from_text(
            text=text_value,
            entities=list(getattr(doc_obj, "ents", []) or []),
            entity_types=entity_types,
            replace_with=config.replace_with,
        )
        if config.regex_patterns:
            for pattern in config.regex_patterns:
                if not pattern:
                    continue
                text_value = re.sub(
                    pattern,
                    config.regex_replace_with,
                    text_value,
                )
        if config.collapse_whitespace:
            text_value = re.sub(r"\s+", " ", text_value).strip()
        processed.append(
            TopicModelingDocument(
                document_id=document.document_id,
                source_item_id=document.source_item_id,
                text=text_value,
            )
        )
        completed += 1
        log_progress()

    report = TopicModelingEntityRemovalReport(
        status=TopicModelingStageStatus.COMPLETE,
        provider=str(config.provider),
        model=str(config.model),
        entity_types=sorted(entity_types),
        input_documents=len(documents),
        output_documents=len(processed),
        regex_patterns=list(config.regex_patterns),
        warnings=[],
        errors=[],
    )
    if cache_path is not None:
        _write_documents_jsonl(cache_path, processed)
    return report, processed


def _write_documents_jsonl(path: Path, documents: List[TopicModelingDocument]) -> None:
    lines = [
        json.dumps(
            {
                "document_id": document.document_id,
                "source_item_id": document.source_item_id,
                "text": document.text,
            },
            ensure_ascii=True,
        )
        for document in documents
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_documents_jsonl(path: Path) -> List[TopicModelingDocument]:
    documents: List[TopicModelingDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            documents.append(
                TopicModelingDocument(
                    document_id=str(payload.get("document_id", "")),
                    source_item_id=str(payload.get("source_item_id", "")),
                    text=str(payload.get("text", "")),
                )
            )
    return documents


def _parse_itemized_response(response_text: str) -> List[str]:
    cleaned = response_text.strip()
    if not cleaned:
        return []
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        unescaped = cleaned.replace('\\"', '"')
        try:
            data = json.loads(unescaped)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return []
        else:
            return []
    items: List[str] = []
    for entry in data:
        if not isinstance(entry, str):
            continue
        cleaned = entry.strip()
        if cleaned:
            items.append(cleaned)
    return items


def _apply_lexical_processing(
    *,
    documents: List[TopicModelingDocument],
    config: TopicModelingLexicalProcessingConfig,
) -> Tuple[TopicModelingLexicalProcessingReport, List[TopicModelingDocument]]:
    if not config.enabled:
        report = TopicModelingLexicalProcessingReport(
            status=TopicModelingStageStatus.SKIPPED,
            input_documents=len(documents),
            output_documents=len(documents),
            lowercase=config.lowercase,
            strip_punctuation=config.strip_punctuation,
            collapse_whitespace=config.collapse_whitespace,
        )
        return report, list(documents)

    processed: List[TopicModelingDocument] = []
    total = len(documents)
    if total <= 50:
        log_interval = 10
    elif total <= 200:
        log_interval = 25
    elif total <= 1000:
        log_interval = 50
    else:
        log_interval = 100
    start_time = time.perf_counter()
    last_log_time = start_time
    completed = 0

    def log_progress() -> None:
        nonlocal last_log_time
        now = time.perf_counter()
        if completed % log_interval == 0 or completed == total or now - last_log_time >= 30.0:
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"[topic-modeling] lexical processing {completed}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
                file=sys.stderr,
            )
            last_log_time = now

    for document in documents:
        text_value = document.text
        if config.lowercase:
            text_value = text_value.lower()
        if config.strip_punctuation:
            text_value = text_value.translate(str.maketrans("", "", string.punctuation))
        if config.collapse_whitespace:
            text_value = re.sub(r"\s+", " ", text_value).strip()
        processed.append(
            TopicModelingDocument(
                document_id=document.document_id,
                source_item_id=document.source_item_id,
                text=text_value,
            )
        )
        completed += 1
        log_progress()

    report = TopicModelingLexicalProcessingReport(
        status=TopicModelingStageStatus.COMPLETE,
        input_documents=len(documents),
        output_documents=len(processed),
        lowercase=config.lowercase,
        strip_punctuation=config.strip_punctuation,
        collapse_whitespace=config.collapse_whitespace,
    )
    return report, processed


def _run_bertopic(
    *,
    documents: List[TopicModelingDocument],
    config: TopicModelingBerTopicConfig,
) -> Tuple[TopicModelingBerTopicReport, List[TopicModelingTopic]]:
    try:
        import importlib

        bertopic_module = importlib.import_module("bertopic")
        if not hasattr(bertopic_module, "BERTopic"):
            raise ImportError("BERTopic class is unavailable")
        BERTopic = bertopic_module.BERTopic
    except ImportError as import_error:
        raise ValueError(
            "BERTopic analysis requires an optional dependency. "
            'Install it with pip install "biblicus[topic-modeling]".'
        ) from import_error

    bertopic_kwargs = dict(config.parameters)
    is_fake = bool(getattr(bertopic_module, "__biblicus_fake__", False))
    if config.vectorizer is not None and "vectorizer_model" not in bertopic_kwargs:
        if is_fake:
            bertopic_kwargs["vectorizer_model"] = None
        else:
            try:
                from sklearn.feature_extraction.text import CountVectorizer
            except ImportError as import_error:
                raise ValueError(
                    "Vectorizer configuration requires scikit-learn. "
                    'Install with pip install "biblicus[topic-modeling]".'
                ) from import_error
            bertopic_kwargs["vectorizer_model"] = CountVectorizer(
                ngram_range=tuple(config.vectorizer.ngram_range),
                stop_words=config.vectorizer.stop_words,
            )

    topic_model = BERTopic(**bertopic_kwargs)
    texts = [document.text for document in documents]
    start_time = time.perf_counter()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(30):
            elapsed = time.perf_counter() - start_time
            print(
                f"[topic-modeling] bertopic fit_transform elapsed={elapsed:.1f}s",
                flush=True,
                file=sys.stderr,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    print(
        f"[topic-modeling] bertopic fit_transform documents={len(texts)}",
        flush=True,
        file=sys.stderr,
    )
    try:
        assignments, _ = topic_model.fit_transform(texts)
    finally:
        stop_event.set()
        thread.join(timeout=1)
    elapsed = time.perf_counter() - start_time
    print(
        f"[topic-modeling] bertopic fit_transform complete elapsed={elapsed:.1f}s",
        flush=True,
        file=sys.stderr,
    )
    assignment_list = list(assignments)
    topic_ids = sorted({int(topic_id) for topic_id in assignment_list})
    topics: List[TopicModelingTopic] = []
    topic_documents = _group_documents_by_topic(documents, assignment_list)

    for topic_id in topic_ids:
        keywords = _resolve_topic_keywords(topic_model=topic_model, topic_id=topic_id)
        label = keywords[0].keyword if keywords else f"Topic {topic_id}"
        doc_entries = topic_documents.get(topic_id, [])
        topics.append(
            TopicModelingTopic(
                topic_id=topic_id,
                label=label,
                label_source=TopicModelingLabelSource.BERTOPIC,
                keywords=keywords,
                document_count=len(doc_entries),
                document_examples=[doc.text for doc in doc_entries[:3]],
                document_ids=[doc.document_id for doc in doc_entries],
            )
        )

    report = TopicModelingBerTopicReport(
        status=TopicModelingStageStatus.COMPLETE,
        topic_count=len(topics),
        document_count=len(documents),
        parameters=dict(config.parameters),
        vectorizer=config.vectorizer,
        warnings=[],
        errors=[],
    )
    return report, topics


def _group_documents_by_topic(
    documents: List[TopicModelingDocument], assignments: List[int]
) -> Dict[int, List[TopicModelingDocument]]:
    grouped: Dict[int, List[TopicModelingDocument]] = {}
    for index, topic_id in enumerate(assignments):
        grouped.setdefault(int(topic_id), []).append(documents[index])
    return grouped


def _resolve_topic_keywords(*, topic_model: Any, topic_id: int) -> List[TopicModelingKeyword]:
    raw_keywords = topic_model.get_topic(topic_id) or []
    return [
        TopicModelingKeyword(keyword=str(entry[0]), score=float(entry[1])) for entry in raw_keywords
    ]


def _apply_llm_fine_tuning(
    *,
    topics: List[TopicModelingTopic],
    documents: List[TopicModelingDocument],
    config: TopicModelingLlmFineTuningConfig,
) -> Tuple[TopicModelingLlmFineTuningReport, List[TopicModelingTopic]]:
    if not config.enabled:
        report = TopicModelingLlmFineTuningReport(
            status=TopicModelingStageStatus.SKIPPED,
            topics_labeled=0,
            warnings=[],
            errors=[],
        )
        return report, topics

    labeled_topics: List[TopicModelingTopic] = []
    errors: List[str] = []
    labeled_count = 0
    total = len(topics)
    if total <= 20:
        log_interval = 1
    elif total <= 50:
        log_interval = 5
    else:
        log_interval = 10
    start_time = time.perf_counter()
    last_log_time = start_time
    topic_documents = {doc.document_id: doc for doc in documents}

    for topic in topics:
        keyword_text = ", ".join(
            [keyword.keyword for keyword in topic.keywords[: config.max_keywords]]
        )
        selected_documents = []
        for doc_id in topic.document_ids[: config.max_documents]:
            doc = topic_documents.get(doc_id)
            if doc is not None:
                selected_documents.append(doc.text)
        documents_text = "\n".join(selected_documents)
        prompt = config.prompt_template.format(
            keywords=keyword_text,
            documents=documents_text,
        )
        label_text = generate_completion(
            client=config.client,
            system_prompt=config.system_prompt,
            user_prompt=prompt,
        ).strip()
        if label_text:
            labeled_topics.append(
                topic.model_copy(
                    update={
                        "label": label_text,
                        "label_source": TopicModelingLabelSource.LLM,
                    }
                )
            )
            labeled_count += 1
        else:
            errors.append(f"LLM fine-tuning returned empty label for topic {topic.topic_id}")
            labeled_topics.append(topic)
        labeled_count = len(labeled_topics)
        now = time.perf_counter()
        if (
            labeled_count % log_interval == 0
            or labeled_count == total
            or now - last_log_time >= 30.0
        ):
            elapsed = now - start_time
            rate = labeled_count / elapsed if elapsed > 0 else 0.0
            print(
                f"[topic-modeling] fine tuning {labeled_count}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                flush=True,
                file=sys.stderr,
            )
            last_log_time = now

    report = TopicModelingLlmFineTuningReport(
        status=TopicModelingStageStatus.COMPLETE,
        topics_labeled=labeled_count,
        warnings=[],
        errors=errors,
    )
    return report, labeled_topics


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


def _write_topic_modeling_output(*, path: Path, output: TopicModelingOutput) -> None:
    path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
