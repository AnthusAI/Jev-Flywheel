"""
Evaluation and benchmarking tools for extraction pipelines.

This module provides tools for quantifying the performance of OCR and other
extraction pipelines against ground truth data.
"""

from biblicus.evaluation.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    CategoryConfig,
    CategoryResult,
)
from biblicus.evaluation.ocr_benchmark import (
    BenchmarkReport,
    OCRBenchmark,
    OCREvaluationResult,
    calculate_character_accuracy,
    calculate_ngram_overlap,
    calculate_word_metrics,
    calculate_word_order_metrics,
)
from biblicus.evaluation.retrieval import _snapshot_artifact_bytes, evaluate_snapshot, load_dataset

__all__ = [
    "BenchmarkReport",
    "OCREvaluationResult",
    "OCRBenchmark",
    "calculate_character_accuracy",
    "calculate_ngram_overlap",
    "calculate_word_metrics",
    "calculate_word_order_metrics",
    "evaluate_snapshot",
    "load_dataset",
    "_snapshot_artifact_bytes",
    # Multi-category benchmark
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "CategoryConfig",
    "CategoryResult",
]
