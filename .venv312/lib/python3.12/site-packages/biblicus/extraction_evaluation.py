"""
Extraction evaluation utilities for Biblicus.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import EXTRACTION_DATASET_SCHEMA_VERSION
from .corpus import Corpus
from .extraction import ExtractionSnapshotManifest
from .models import CatalogItem
from .time import utc_now_iso


class ExtractionEvaluationItem(BaseModel):
    """
    Dataset item for extraction evaluation.

    :ivar item_id: Optional item identifier.
    :vartype item_id: str or None
    :ivar source_uri: Optional source uniform resource identifier.
    :vartype source_uri: str or None
    :ivar expected_text: Expected extracted text.
    :vartype expected_text: str
    :ivar kind: Label kind (gold or synthetic).
    :vartype kind: str
    """

    model_config = ConfigDict(extra="forbid")

    item_id: Optional[str] = None
    source_uri: Optional[str] = None
    expected_text: str
    kind: str = Field(default="gold")

    @model_validator(mode="after")
    def _require_locator(self) -> "ExtractionEvaluationItem":
        if not self.item_id and not self.source_uri:
            raise ValueError("Evaluation items must include item_id or source_uri")
        return self


class ExtractionEvaluationDataset(BaseModel):
    """
    Dataset for extraction evaluation.

    :ivar schema_version: Dataset schema version.
    :vartype schema_version: int
    :ivar name: Dataset name.
    :vartype name: str
    :ivar description: Optional description.
    :vartype description: str or None
    :ivar items: Labeled evaluation items.
    :vartype items: list[ExtractionEvaluationItem]
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    name: str
    description: Optional[str] = None
    items: List[ExtractionEvaluationItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_schema_version(self) -> "ExtractionEvaluationDataset":
        if self.schema_version != EXTRACTION_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported extraction dataset schema version: {self.schema_version}"
            )
        return self


class ExtractionEvaluationItemReport(BaseModel):
    """
    Per-item report for extraction evaluation.

    :ivar item_id: Item identifier.
    :vartype item_id: str
    :ivar source_uri: Source uniform resource identifier.
    :vartype source_uri: str
    :ivar expected_text: Expected text from the dataset.
    :vartype expected_text: str
    :ivar extracted_text: Extracted text when available.
    :vartype extracted_text: str or None
    :ivar coverage_status: Coverage status (present, empty, missing).
    :vartype coverage_status: str
    :ivar extraction_status: Extraction status from the run (extracted, skipped, errored, missing).
    :vartype extraction_status: str
    :ivar similarity_score: Similarity score between expected and extracted text.
    :vartype similarity_score: float
    :ivar kind: Label kind from the dataset.
    :vartype kind: str
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    source_uri: str
    expected_text: str
    extracted_text: Optional[str] = None
    coverage_status: str
    extraction_status: str
    similarity_score: float
    kind: str


class ExtractionEvaluationResult(BaseModel):
    """
    Result bundle for an extraction evaluation.

    :ivar dataset: Dataset metadata.
    :vartype dataset: dict[str, object]
    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    :ivar snapshot_id: Extraction snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration_id: Extraction configuration identifier.
    :vartype configuration_id: str
    :ivar configuration_name: Extraction configuration name.
    :vartype configuration_name: str
    :ivar evaluated_at: International Organization for Standardization 8601 timestamp.
    :vartype evaluated_at: str
    :ivar metrics: Evaluation metrics for coverage and accuracy.
    :vartype metrics: dict[str, float]
    :ivar items: Per-item evaluation reports.
    :vartype items: list[ExtractionEvaluationItemReport]
    """

    model_config = ConfigDict(extra="forbid")

    dataset: Dict[str, object]
    extractor_id: str
    snapshot_id: str
    configuration_id: str
    configuration_name: str
    evaluated_at: str
    metrics: Dict[str, float]
    items: List[ExtractionEvaluationItemReport]


def load_extraction_dataset(path: Path) -> ExtractionEvaluationDataset:
    """
    Load an extraction evaluation dataset from JavaScript Object Notation.

    :param path: Path to the dataset file.
    :type path: Path
    :return: Parsed extraction evaluation dataset.
    :rtype: ExtractionEvaluationDataset
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid extraction dataset") from exc
    return ExtractionEvaluationDataset.model_validate(data)


def evaluate_extraction_snapshot(
    *,
    corpus: Corpus,
    snapshot: ExtractionSnapshotManifest,
    extractor_id: str,
    dataset: ExtractionEvaluationDataset,
) -> ExtractionEvaluationResult:
    """
    Evaluate an extraction snapshot against a dataset.

    :param corpus: Corpus associated with the snapshot.
    :type corpus: Corpus
    :param snapshot: Extraction snapshot manifest.
    :type snapshot: ExtractionSnapshotManifest
    :param extractor_id: Extractor identifier for the snapshot.
    :type extractor_id: str
    :param dataset: Extraction evaluation dataset.
    :type dataset: ExtractionEvaluationDataset
    :return: Extraction evaluation result bundle.
    :rtype: ExtractionEvaluationResult
    """
    catalog = corpus.load_catalog()
    item_index = {item.item_id: item for item in snapshot.items}
    coverage_present = 0
    coverage_empty = 0
    coverage_missing = 0
    processable = 0
    similarity_scores: List[float] = []
    item_reports: List[ExtractionEvaluationItemReport] = []

    for dataset_item in dataset.items:
        item_id = _resolve_item_id(dataset_item, catalog_items=catalog.items)
        catalog_item = catalog.items.get(item_id)
        if catalog_item is None:
            raise ValueError(f"Unknown item identifier: {item_id}")
        extraction_item = item_index.get(item_id)
        extraction_status = extraction_item.status if extraction_item else "missing"
        if extraction_status != "errored" and extraction_status != "missing":
            processable += 1

        extracted_text = corpus.read_extracted_text(
            extractor_id=extractor_id, snapshot_id=snapshot.snapshot_id, item_id=item_id
        )
        coverage_status = _coverage_status(extracted_text)
        if coverage_status == "present":
            coverage_present += 1
        elif coverage_status == "empty":
            coverage_empty += 1
        else:
            coverage_missing += 1

        similarity_score = _similarity_score(
            expected_text=dataset_item.expected_text, extracted_text=extracted_text
        )
        similarity_scores.append(similarity_score)
        item_reports.append(
            ExtractionEvaluationItemReport(
                item_id=item_id,
                source_uri=catalog_item.source_uri,
                expected_text=dataset_item.expected_text,
                extracted_text=extracted_text,
                coverage_status=coverage_status,
                extraction_status=extraction_status,
                similarity_score=similarity_score,
                kind=dataset_item.kind,
            )
        )

    total_items = max(len(dataset.items), 1)
    average_similarity = sum(similarity_scores) / total_items if similarity_scores else 0.0
    metrics = {
        "coverage_present": float(coverage_present),
        "coverage_empty": float(coverage_empty),
        "coverage_missing": float(coverage_missing),
        "processable_fraction": processable / total_items,
        "average_similarity": average_similarity,
    }
    dataset_meta = {
        "name": dataset.name,
        "description": dataset.description,
        "items": len(dataset.items),
    }
    return ExtractionEvaluationResult(
        dataset=dataset_meta,
        extractor_id=extractor_id,
        snapshot_id=snapshot.snapshot_id,
        configuration_id=snapshot.configuration.configuration_id,
        configuration_name=snapshot.configuration.name,
        evaluated_at=utc_now_iso(),
        metrics=metrics,
        items=item_reports,
    )


def write_extraction_evaluation_result(
    *, corpus: Corpus, snapshot_id: str, result: ExtractionEvaluationResult
) -> Path:
    """
    Persist extraction evaluation output under the corpus.

    :param corpus: Corpus associated with the evaluation.
    :type corpus: Corpus
    :param snapshot_id: Extraction snapshot identifier.
    :type snapshot_id: str
    :param result: Evaluation result to write.
    :type result: ExtractionEvaluationResult
    :return: Output path.
    :rtype: Path
    """
    output_dir = corpus.analysis_dir / "evaluation" / "extraction" / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.json"
    output_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def _resolve_item_id(
    dataset_item: ExtractionEvaluationItem, *, catalog_items: Dict[str, CatalogItem]
) -> str:
    if dataset_item.item_id:
        return dataset_item.item_id
    source_uri = dataset_item.source_uri
    if not source_uri:
        raise ValueError("Evaluation item is missing item_id and source_uri")
    for item_id, catalog_item in catalog_items.items():
        if getattr(catalog_item, "source_uri", None) == source_uri:
            return item_id
    raise ValueError(f"Unknown source uniform resource identifier: {source_uri}")


def _coverage_status(extracted_text: Optional[str]) -> str:
    if extracted_text is None:
        return "missing"
    if extracted_text.strip():
        return "present"
    return "empty"


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _similarity_score(*, expected_text: str, extracted_text: Optional[str]) -> float:
    if extracted_text is None:
        return 0.0
    expected = _normalize_text(expected_text)
    actual = _normalize_text(extracted_text)
    if not expected and not actual:
        return 1.0
    return SequenceMatcher(None, expected, actual).ratio()
