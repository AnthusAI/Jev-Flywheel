"""
Multi-category benchmark runner for the Biblicus Document Understanding Benchmark.

This module orchestrates benchmarking across multiple document categories:
- Forms (FUNSD)
- Academic papers (Scanned ArXiv)
- Receipts (SROIE)

Each category has its own primary metric and evaluation approach, with results
aggregated into a unified benchmark report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from biblicus.corpus import Corpus
from biblicus.evaluation.ocr_benchmark import OCRBenchmark


@dataclass
class CategoryConfig:
    """Configuration for a single benchmark category."""

    name: str
    dataset: str
    primary_metric: str
    pipelines: List[object] = field(default_factory=list)
    corpus_path: Optional[Path] = None
    ground_truth_subdir: Optional[str] = None
    subset_size: Optional[int] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for a complete benchmark run."""

    benchmark_name: str
    categories: Dict[str, CategoryConfig]
    pipelines: List[Path]
    aggregate_weights: Dict[str, float]
    output_dir: Path = Path("results")

    @classmethod
    def load(cls, config_path: Path) -> "BenchmarkConfig":
        """
        Load benchmark configuration from YAML file.

        :param config_path: Path to configuration file.
        :type config_path: Path
        :return: Loaded configuration.
        :rtype: BenchmarkConfig
        """
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        categories = {}
        for name, cat_data in data.get("categories", {}).items():
            categories[name] = CategoryConfig(
                name=name,
                dataset=cat_data["dataset"],
                corpus_path=Path(cat_data.get("corpus_path", f"corpora/{cat_data['dataset']}_benchmark")),
                ground_truth_subdir=cat_data.get("ground_truth_subdir", f"{cat_data['dataset']}_ground_truth"),
                primary_metric=cat_data.get("primary_metric", "f1_score"),
                subset_size=cat_data.get("subset_size"),
                tags=cat_data.get("tags", []),
            )

        pipelines = [Path(p) for p in data.get("pipelines", [])]

        aggregate_weights = data.get("aggregate_weights", {
            "forms": 0.40,
            "academic": 0.35,
            "receipts": 0.25,
        })

        return cls(
            benchmark_name=data.get("benchmark_name", "standard"),
            categories=categories,
            pipelines=pipelines,
            aggregate_weights=aggregate_weights,
            output_dir=Path(data.get("output_dir", "results")),
        )


@dataclass
class CategoryResult:
    """Results for a single category."""

    category_name: str
    dataset: str
    documents_evaluated: int
    pipelines: List[Dict[str, Any]]
    best_pipeline: str
    best_score: float
    primary_metric: str
    primary_score: float
    processing_time_seconds: float


def run_category(output_dir: Path, cat_config: CategoryConfig, runner: object) -> CategoryResult:
    """
    Execute a single benchmark category using a provided runner.

    This utility is intentionally lightweight so tests can exercise the
    primary-metric selection logic without needing full corpus assets.

    :param output_dir: Directory where intermediate artifacts may be written.
    :type output_dir: Path
    :param cat_config: Parsed category configuration.
    :type cat_config: CategoryConfig
    :param runner: Object exposing ``run_pipeline(name)`` returning metrics.
    :type runner: object
    :return: CategoryResult summarizing the best pipeline.
    :rtype: CategoryResult
    """
    _ = output_dir
    start_time = time.time()
    pipelines: List[Dict[str, Any]] = []
    best_pipeline = ""
    best_score = float("-inf")
    documents_evaluated = 0

    for pipeline in cat_config.pipelines:
        pipeline_name = getattr(pipeline, "name", str(pipeline))
        metrics_obj = runner.run_pipeline(pipeline_name)
        metrics = {
            "f1": getattr(metrics_obj, "avg_f1", None),
            "recall": getattr(metrics_obj, "avg_recall", None),
            "precision": getattr(metrics_obj, "avg_precision", None),
            "wer": getattr(metrics_obj, "avg_word_error_rate", None),
            "lcs_ratio": getattr(metrics_obj, "avg_lcs_ratio", None),
            "bigram_overlap": getattr(metrics_obj, "avg_bigram_overlap", None),
            "sequence_accuracy": getattr(metrics_obj, "avg_sequence_accuracy", None),
        }
        documents_evaluated = getattr(metrics_obj, "total_documents", documents_evaluated)
        pipelines.append({"name": pipeline_name, "metrics": metrics})

        primary_value = metrics.get(cat_config.primary_metric)
        if primary_value is None:
            primary_value = getattr(metrics_obj, f"avg_{cat_config.primary_metric}", None)
        if primary_value is None:
            continue
        if primary_value > best_score:
            best_score = primary_value
            best_pipeline = pipeline_name

    processing_time = time.time() - start_time

    if best_score == float("-inf"):
        best_score = 0.0

    return CategoryResult(
        category_name=cat_config.name,
        dataset=cat_config.dataset,
        documents_evaluated=documents_evaluated,
        pipelines=pipelines,
        best_pipeline=best_pipeline,
        best_score=best_score,
        primary_metric=cat_config.primary_metric,
        primary_score=best_score,
        processing_time_seconds=processing_time,
    )


@dataclass
class BenchmarkResult:
    """Complete benchmark results across all categories."""

    benchmark_version: str = "1.0.0"
    benchmark_name: str = ""
    timestamp: str = ""
    categories: Dict[str, CategoryResult] = field(default_factory=dict)
    aggregate: Dict[str, float] = field(default_factory=dict)
    recommendations: Dict[str, str] = field(default_factory=dict)
    total_documents: int = 0
    total_processing_time_seconds: float = 0.0

    def to_json(self, path: Path) -> None:
        """Export results to JSON file."""
        data = {
            "benchmark_version": self.benchmark_version,
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "total_documents": self.total_documents,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "categories": {},
            "aggregate": self.aggregate,
            "recommendations": self.recommendations,
        }

        for cat_name, cat_result in self.categories.items():
            data["categories"][cat_name] = {
                "dataset": cat_result.dataset,
                "documents_evaluated": cat_result.documents_evaluated,
                "primary_metric": cat_result.primary_metric,
                "best_pipeline": cat_result.best_pipeline,
                "best_score": cat_result.best_score,
                "processing_time_seconds": cat_result.processing_time_seconds,
                "pipelines": cat_result.pipelines,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_markdown(self, path: Path) -> None:
        """Export results to Markdown file."""
        lines = [
            "# Biblicus Document Understanding Benchmark Results",
            "",
            f"**Benchmark:** {self.benchmark_name}",
            f"**Date:** {self.timestamp}",
            f"**Total Documents:** {self.total_documents}",
            f"**Processing Time:** {self.total_processing_time_seconds:.1f}s",
            "",
            "## Executive Summary",
            "",
            "| Category | Dataset | Docs | Best Pipeline | Score | Metric |",
            "|----------|---------|------|---------------|-------|--------|",
        ]

        for cat_name, cat_result in self.categories.items():
            lines.append(
                f"| {cat_name.title()} | {cat_result.dataset} | {cat_result.documents_evaluated} | "
                f"{cat_result.best_pipeline} | {cat_result.best_score:.3f} | {cat_result.primary_metric} |"
            )

        lines.extend([
            "",
            "## Aggregate Score",
            "",
            f"**Weighted Score:** {self.aggregate.get('weighted_score', 0):.3f}",
            "",
            "**Weights:**",
        ])

        for cat, weight in self.aggregate.get("weights", {}).items():
            lines.append(f"- {cat.title()}: {weight:.0%}")

        lines.extend([
            "",
            "## Recommendations",
            "",
        ])

        for rec_type, pipeline in self.recommendations.items():
            rec_label = rec_type.replace("_", " ").title()
            lines.append(f"- **{rec_label}:** {pipeline}")

        lines.extend([
            "",
            "## Category Details",
            "",
        ])

        for cat_name, cat_result in self.categories.items():
            lines.extend([
                f"### {cat_name.title()} ({cat_result.dataset})",
                "",
                f"Primary Metric: {cat_result.primary_metric}",
                "",
                "| Pipeline | F1 | Recall | Precision | WER | LCS |",
                "|----------|-----|--------|-----------|-----|-----|",
            ])

            for pipeline in cat_result.pipelines:
                metrics = pipeline.get("metrics", {})
                lines.append(
                    f"| {pipeline['name']} | "
                    f"{metrics.get('f1', 0):.3f} | "
                    f"{metrics.get('recall', 0):.3f} | "
                    f"{metrics.get('precision', 0):.3f} | "
                    f"{metrics.get('wer', 0):.3f} | "
                    f"{metrics.get('lcs_ratio', 0):.3f} |"
                )

            lines.append("")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def print_summary(self) -> None:
        """Print summary to console."""
        print("\n" + "=" * 70)
        print(f"BIBLICUS DOCUMENT UNDERSTANDING BENCHMARK: {self.benchmark_name}")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Documents: {self.total_documents}")
        print(f"Processing Time: {self.total_processing_time_seconds:.1f}s")
        print()

        print("Category Results:")
        print("-" * 70)
        for cat_name, cat_result in self.categories.items():
            print(f"  {cat_name.title():12} | {cat_result.dataset:15} | "
                  f"{cat_result.documents_evaluated:4} docs | "
                  f"Best: {cat_result.best_pipeline} ({cat_result.best_score:.3f} {cat_result.primary_metric})")

        print()
        print(f"Aggregate Score: {self.aggregate.get('weighted_score', 0):.3f}")
        print()

        print("Recommendations:")
        for rec_type, pipeline in self.recommendations.items():
            print(f"  {rec_type.replace('_', ' ').title():20}: {pipeline}")

        print("=" * 70)


class BenchmarkRunner:
    """
    Orchestrates multi-category benchmarking.

    Usage:
        config = BenchmarkConfig.load("configs/benchmark/standard.yaml")
        runner = BenchmarkRunner(config)
        results = runner.run_all()
        results.to_json(Path("results/benchmark.json"))
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark runner.

        :param config: Benchmark configuration.
        :type config: BenchmarkConfig
        """
        self.config = config

    def run_all(self) -> BenchmarkResult:
        """
        Run benchmark across all configured categories.

        :return: Complete benchmark results.
        :rtype: BenchmarkResult
        """
        result = BenchmarkResult(
            benchmark_name=self.config.benchmark_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        start_time = time.time()

        for cat_name, cat_config in self.config.categories.items():
            print(f"\n{'='*60}")
            print(f"Running {cat_name.upper()} benchmark ({cat_config.dataset})...")
            print(f"{'='*60}")

            try:
                cat_result = self.run_category(cat_config)
                result.categories[cat_name] = cat_result
                result.total_documents += cat_result.documents_evaluated
            except Exception as e:
                print(f"  ERROR: Failed to run {cat_name} benchmark: {e}")
                continue

        result.total_processing_time_seconds = time.time() - start_time

        # Calculate aggregate score
        result.aggregate = self._calculate_aggregate(result.categories)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result.categories)

        return result

    def run_category(self, cat_config: CategoryConfig) -> CategoryResult:
        """
        Run benchmark for a single category.

        :param cat_config: Category configuration.
        :type cat_config: CategoryConfig
        :return: Category results.
        :rtype: CategoryResult
        """
        start_time = time.time()

        # Open corpus
        if not cat_config.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {cat_config.corpus_path}")

        corpus = Corpus.open(cat_config.corpus_path)

        # Get ground truth directory
        gt_dir = corpus.meta_dir / cat_config.ground_truth_subdir
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

        # Initialize benchmark
        benchmark = OCRBenchmark(corpus)

        pipeline_results: List[Dict[str, Any]] = []
        best_pipeline = ""
        best_score = 0.0

        for pipeline_path in self.config.pipelines:
            pipeline_name = pipeline_path.stem

            print(f"\n  Testing pipeline: {pipeline_name}")

            try:
                # Load pipeline config
                with open(pipeline_path, "r", encoding="utf-8") as f:
                    pipeline_config = yaml.safe_load(f)

                # Run extraction
                extractor_id = pipeline_config.get("extractor_id", "pipeline")
                config = pipeline_config.get("config", {})

                # Build extraction snapshot
                snapshot = corpus.extract(extractor_id=extractor_id, config=config)

                # Evaluate against ground truth
                report = benchmark.evaluate_extraction(
                    snapshot_reference=snapshot.snapshot_id,
                    ground_truth_dir=gt_dir,
                )

                # Extract metrics
                metrics = {
                    "f1": report.avg_f1,
                    "recall": report.avg_recall,
                    "precision": report.avg_precision,
                    "wer": report.avg_word_error_rate,
                    "lcs_ratio": report.avg_lcs_ratio,
                    "bigram_overlap": report.avg_bigram_overlap,
                    "sequence_accuracy": report.avg_sequence_accuracy,
                }

                pipeline_results.append({
                    "name": pipeline_name,
                    "metrics": metrics,
                    "documents_evaluated": report.total_documents,
                })

                # Check if this is the best pipeline for primary metric
                primary_score = metrics.get(cat_config.primary_metric, metrics.get("f1", 0))
                if primary_score > best_score:
                    best_score = primary_score
                    best_pipeline = pipeline_name

                print(f"    {cat_config.primary_metric}: {primary_score:.3f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        processing_time = time.time() - start_time

        return CategoryResult(
            category_name=cat_config.name,
            dataset=cat_config.dataset,
            documents_evaluated=pipeline_results[0]["documents_evaluated"] if pipeline_results else 0,
            pipelines=pipeline_results,
            best_pipeline=best_pipeline,
            best_score=best_score,
            primary_metric=cat_config.primary_metric,
            primary_score=best_score,
            processing_time_seconds=processing_time,
        )

    def _calculate_aggregate(self, categories: Dict[str, CategoryResult]) -> Dict[str, float]:
        """Calculate weighted aggregate score."""
        weighted_sum = 0.0
        total_weight = 0.0

        weights = {}

        for cat_name, cat_result in categories.items():
            weight = self.config.aggregate_weights.get(cat_name, 0.0)
            if weight > 0:
                weighted_sum += cat_result.best_score * weight
                total_weight += weight
                weights[cat_name] = weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "weighted_score": weighted_score,
            "weights": weights,
        }

    def _generate_recommendations(self, categories: Dict[str, CategoryResult]) -> Dict[str, str]:
        """Generate pipeline recommendations based on results."""
        recommendations = {}

        # Best overall (highest aggregate across categories)
        pipeline_scores: Dict[str, List[float]] = {}
        for cat_result in categories.values():
            for pipeline in cat_result.pipelines:
                name = pipeline["name"]
                if name not in pipeline_scores:
                    pipeline_scores[name] = []
                pipeline_scores[name].append(pipeline["metrics"].get("f1", 0))

        if pipeline_scores:
            best_overall = max(pipeline_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
            recommendations["best_overall"] = best_overall[0]

        # Best for layout (highest LCS ratio)
        best_lcs = ""
        best_lcs_score = 0.0
        for cat_result in categories.values():
            for pipeline in cat_result.pipelines:
                lcs = pipeline["metrics"].get("lcs_ratio", 0)
                if lcs > best_lcs_score:
                    best_lcs_score = lcs
                    best_lcs = pipeline["name"]
        if best_lcs:
            recommendations["best_for_layout"] = best_lcs

        # Best for recall
        best_recall = ""
        best_recall_score = 0.0
        for cat_result in categories.values():
            for pipeline in cat_result.pipelines:
                recall = pipeline["metrics"].get("recall", 0)
                if recall > best_recall_score:
                    best_recall_score = recall
                    best_recall = pipeline["name"]
        if best_recall:
            recommendations["best_for_completeness"] = best_recall

        # Best for precision
        best_precision = ""
        best_precision_score = 0.0
        for cat_result in categories.values():
            for pipeline in cat_result.pipelines:
                precision = pipeline["metrics"].get("precision", 0)
                if precision > best_precision_score:
                    best_precision_score = precision
                    best_precision = pipeline["name"]
        if best_precision:
            recommendations["best_for_accuracy"] = best_precision

        return recommendations
