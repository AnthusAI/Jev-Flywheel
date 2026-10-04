"""
Pipeline recipe execution for Biblicus.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
from pydantic import ValidationError

from .analysis import get_analysis_backend
from .collections import load_collection_config, pull_collection
from .configuration import load_configuration_view
from .corpus import Corpus
from .extraction import build_extraction_snapshot
from .models import (
    ExtractionSnapshotReference,
    PipelineAnalysisConfig,
    PipelineRecipeConfig,
    RetrievalSnapshot,
)
from .retrievers import get_retriever
from .workflow import build_default_handler_registry, build_plan_for_index


class PipelineRunResult:
    """
    Summary of a pipeline recipe run.

    :ivar corpora: List of corpus paths targeted by the recipe.
    :vartype corpora: list[str]
    :ivar extraction_snapshot_ids: Extraction snapshot identifiers produced.
    :vartype extraction_snapshot_ids: list[str]
    :ivar retrieval_snapshot_ids: Retrieval snapshot identifiers produced.
    :vartype retrieval_snapshot_ids: list[str]
    """

    def __init__(self) -> None:
        self.corpora: List[str] = []
        self.extraction_snapshot_ids: List[str] = []
        self.retrieval_snapshot_ids: List[str] = []


def load_pipeline_recipe(recipe_path: Path) -> PipelineRecipeConfig:
    """
    Load a pipeline recipe from a YAML file.

    :param recipe_path: Pipeline recipe path.
    :type recipe_path: Path
    :return: Parsed pipeline recipe configuration.
    :rtype: PipelineRecipeConfig
    :raises FileNotFoundError: If the recipe file is missing.
    :raises ValueError: If the recipe is invalid.
    """
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Pipeline recipe not found: {recipe_path}")
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline recipe must be a mapping/object")
    return PipelineRecipeConfig.model_validate(raw)


def run_pipeline_recipe(recipe_path: Path) -> PipelineRunResult:
    """
    Execute a pipeline recipe.

    :param recipe_path: Pipeline recipe path.
    :type recipe_path: Path
    :return: Pipeline run summary.
    :rtype: PipelineRunResult
    """
    recipe = load_pipeline_recipe(recipe_path)
    if recipe.mirror is not None:
        collection_root = _resolve_collection_root(recipe.mirror.collection)
        pull_collection(collection_root)

    corpora = _resolve_target_corpora(recipe)
    result = PipelineRunResult()

    for corpus_path in corpora:
        corpus = Corpus.open(corpus_path)
        result.corpora.append(str(corpus.root))
        extraction_ref = None
        if recipe.extraction is not None:
            extraction_ref = _run_extraction(corpus, recipe.extraction)
            result.extraction_snapshot_ids.append(extraction_ref.snapshot_id)
        if recipe.retrieval is not None:
            snapshot = _run_retrieval(corpus, recipe.retrieval)
            result.retrieval_snapshot_ids.append(snapshot.snapshot_id)
        if recipe.analysis:
            _run_analysis(corpus, recipe.analysis, extraction_ref)

    return result


def _resolve_collection_root(collection_ref: str) -> Path:
    candidate = Path(collection_ref)
    if candidate.exists():
        return candidate
    return Path("collections") / collection_ref


def _resolve_target_corpora(recipe: PipelineRecipeConfig) -> List[Path]:
    selector = recipe.corpus
    if selector.path:
        return [Path(selector.path)]
    collection_root = _resolve_collection_root(selector.collection or "")
    collection_config = load_collection_config(collection_root)
    corpus_root = _resolve_collection_corpus_root(collection_root, collection_config.corpus_root)
    if not corpus_root.exists():
        raise FileNotFoundError(f"Collection corpus root not found: {corpus_root}")
    patterns = selector.selector or "*"
    paths = []
    for child in corpus_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if fnmatch.fnmatch(child.name, patterns):
            paths.append(child)
    return paths


def _resolve_collection_corpus_root(collection_root: Path, corpus_root: str) -> Path:
    base = collection_root.parent
    if base.name == "collections":
        base = base.parent
    root = Path(corpus_root)
    if root.is_absolute():
        return root
    return (base / root).resolve()


def _run_extraction(corpus: Corpus, extraction_config) -> ExtractionSnapshotReference:
    configuration_data = load_configuration_view(
        [str(extraction_config.recipe)],
        configuration_label="Extraction recipe",
        mapping_error_message="Extraction recipe must be a mapping/object",
    )
    extractor_id, configuration, max_workers = _normalize_extraction_configuration(
        configuration_data
    )
    configuration_name = Path(extraction_config.recipe).stem
    manifest = build_extraction_snapshot(
        corpus,
        extractor_id=extractor_id,
        configuration_name=configuration_name,
        configuration=configuration,
        max_workers=max_workers or 1,
    )
    return ExtractionSnapshotReference(
        extractor_id=extractor_id, snapshot_id=manifest.snapshot_id
    )


def _normalize_extraction_configuration(
    configuration_data: Dict[str, object],
) -> tuple[str, Dict[str, object], Optional[int]]:
    extractor_id = configuration_data.get("extractor_id", "pipeline")
    configuration = configuration_data.get("configuration", {})
    max_workers = configuration_data.get("max_workers")
    if configuration is None:
        configuration = {}
    if not isinstance(configuration, dict):
        raise ValueError("Extraction configuration must be a mapping/object")
    if not isinstance(extractor_id, str) or not extractor_id.strip():
        raise ValueError("Extraction configuration must include a non-empty extractor_id")
    extractor_id = extractor_id.strip()
    if max_workers is not None:
        if isinstance(max_workers, bool):
            raise ValueError("Extraction configuration max_workers must be an integer")
        try:
            max_workers = int(max_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("Extraction configuration max_workers must be an integer") from exc
        if max_workers < 1:
            raise ValueError("Extraction configuration max_workers must be >= 1")
    if extractor_id != "pipeline":
        return (
            "pipeline",
            {"stages": [{"extractor_id": extractor_id, "config": configuration}]},
            max_workers,
        )
    return "pipeline", configuration, max_workers


def _run_retrieval(corpus: Corpus, retrieval_config) -> RetrievalSnapshot:
    base_config: Dict[str, object] = load_configuration_view(
        [str(retrieval_config.configuration)],
        configuration_label="Retrieval configuration",
        mapping_error_message="Retrieval configuration must be a mapping/object",
    )
    retriever = get_retriever(retrieval_config.retriever)

    plan = build_plan_for_index(
        corpus,
        retriever_id=retrieval_config.retriever,
        pipeline_config=None,
        index_config=base_config,
        load_handler_available=False,
    )
    if plan.status == "blocked":
        raise ValueError(plan.root.reason or "Retrieval dependencies blocked")
    if plan.status != "complete":
        registry = build_default_handler_registry(corpus)
        plan.execute(mode="auto", handler_registry=registry)

    snapshot = retriever.build_snapshot(
        corpus,
        configuration_name=Path(retrieval_config.configuration).stem,
        configuration=base_config,
    )
    return snapshot


def _run_analysis(
    corpus: Corpus,
    analysis_configs: Iterable[PipelineAnalysisConfig],
    extraction_snapshot: Optional[ExtractionSnapshotReference],
) -> None:
    resolved_snapshot = extraction_snapshot or corpus.latest_extraction_snapshot_reference()
    if resolved_snapshot is None:
        raise ValueError("Analysis requires an extraction snapshot")
    for analysis in analysis_configs:
        configuration_data: Dict[str, object] = {}
        if analysis.configuration:
            configuration_data = load_configuration_view(
                [str(analysis.configuration)],
                configuration_label="Analysis configuration",
                mapping_error_message="Analysis configuration must be a mapping/object",
            )
        backend = get_analysis_backend(analysis.kind)
        try:
            backend.run_analysis(
                corpus,
                configuration_name=Path(analysis.configuration).stem,
                configuration=configuration_data,
                extraction_snapshot=resolved_snapshot,
            )
        except ValidationError as exc:
            raise ValueError(f"Invalid analysis configuration: {exc}") from exc
