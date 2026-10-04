"""
OCR benchmarking and evaluation system.

Provides comprehensive tools for evaluating OCR pipeline performance against
ground truth data with detailed per-document and aggregate metrics.
"""

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from biblicus import Corpus


def normalize_words(text: str) -> List[str]:
    """
    Normalize text into words for comparison.

    - Lowercases
    - Removes punctuation
    - Filters words > 2 characters
    """
    # Lowercase and split
    words = text.lower().split()

    # Remove punctuation and filter short words
    cleaned = []
    for word in words:
        # Remove all non-alphanumeric
        clean = re.sub(r'[^a-z0-9]', '', word)
        if len(clean) > 2:
            cleaned.append(clean)

    return cleaned


def calculate_word_metrics(ground_truth: str, extracted: str) -> Dict[str, Any]:
    """
    Calculate word-level precision, recall, and F1 score.

    Compares word sets after normalization (lowercase, remove punctuation).

    Args:
        ground_truth: Expected text
        extracted: Actual OCR output

    Returns:
        Dictionary with precision, recall, f1_score, and counts
    """
    gt_words = normalize_words(ground_truth)
    ex_words = normalize_words(extracted)

    gt_set = set(gt_words)
    ex_set = set(ex_words)

    true_positives = len(gt_set & ex_set)
    false_positives = len(ex_set - gt_set)
    false_negatives = len(gt_set - ex_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "word_count_gt": len(gt_words),
        "word_count_ocr": len(ex_words),
    }


def calculate_character_accuracy(ground_truth: str, extracted: str) -> float:
    """
    Calculate character-level accuracy using edit distance.

    Uses Levenshtein distance to compute how similar the strings are.
    Returns 1.0 - (distance / max_length).

    Args:
        ground_truth: Expected text
        extracted: Actual OCR output

    Returns:
        Accuracy between 0.0 and 1.0
    """
    try:
        import editdistance
        distance = editdistance.eval(ground_truth, extracted)
        max_len = max(len(ground_truth), len(extracted))
        if max_len == 0:
            return 1.0
        accuracy = 1.0 - (distance / max_len)
        return max(0.0, accuracy)  # Clamp to [0, 1]
    except ImportError:
        # editdistance not installed, use simple character overlap
        gt_chars = set(ground_truth.lower())
        ex_chars = set(extracted.lower())
        if not gt_chars:
            return 1.0 if not ex_chars else 0.0
        overlap = len(gt_chars & ex_chars) / len(gt_chars)
        return overlap


def calculate_word_order_metrics(ground_truth: str, extracted: str) -> Dict[str, Any]:
    """
    Calculate order-aware metrics that measure reading sequence quality.

    These metrics are critical for evaluating layout-aware OCR where the goal
    is to preserve correct reading order (e.g., left column before right column).

    Metrics:
    - Word Error Rate (WER): Edit distance on word sequences (insertions, deletions, substitutions)
    - Sequence accuracy: What % of word sequences match exactly
    - Longest Common Subsequence (LCS): Longest sequence of words in correct order
    - Normalized edit distance: Word-level Levenshtein distance normalized by length

    Args:
        ground_truth: Expected text in correct reading order
        extracted: Actual OCR output

    Returns:
        Dictionary with order-aware metrics
    """
    gt_words = normalize_words(ground_truth)
    ex_words = normalize_words(extracted)

    if not gt_words and not ex_words:
        return {
            'word_error_rate': 0.0,
            'sequence_accuracy': 1.0,
            'lcs_ratio': 1.0,
            'normalized_edit_distance': 0.0,
        }

    if not gt_words or not ex_words:
        return {
            'word_error_rate': 1.0,
            'sequence_accuracy': 0.0,
            'lcs_ratio': 0.0,
            'normalized_edit_distance': 1.0,
        }

    # Calculate Word Error Rate (WER) using edit distance on word sequences
    try:
        import editdistance
        edit_dist = editdistance.eval(gt_words, ex_words)
        wer = edit_dist / len(gt_words)
        normalized_edit_dist = edit_dist / max(len(gt_words), len(ex_words))
    except ImportError:
        # Fallback: simple implementation
        edit_dist = _simple_edit_distance(gt_words, ex_words)
        wer = edit_dist / len(gt_words)
        normalized_edit_dist = edit_dist / max(len(gt_words), len(ex_words))

    # Calculate Longest Common Subsequence (LCS)
    lcs_length = _longest_common_subsequence(gt_words, ex_words)
    lcs_ratio = lcs_length / len(gt_words) if gt_words else 0.0

    # Sequence accuracy: ratio of correct words in correct positions
    matches = sum(1 for i, word in enumerate(gt_words) if i < len(ex_words) and ex_words[i] == word)
    sequence_accuracy = matches / len(gt_words)

    return {
        'word_error_rate': wer,
        'sequence_accuracy': sequence_accuracy,
        'lcs_ratio': lcs_ratio,
        'normalized_edit_distance': normalized_edit_dist,
    }


def calculate_ngram_overlap(ground_truth: str, extracted: str, n: int = 2) -> float:
    """
    Calculate n-gram overlap to measure local word ordering.

    N-grams capture short sequences of words. High n-gram overlap means
    the extracted text preserves local word ordering, even if global order differs.

    Args:
        ground_truth: Expected text
        extracted: Actual OCR output
        n: N-gram size (default 2 for bigrams)

    Returns:
        N-gram overlap ratio (0.0 to 1.0)
    """
    gt_words = normalize_words(ground_truth)
    ex_words = normalize_words(extracted)

    if len(gt_words) < n or len(ex_words) < n:
        return 0.0

    # Create n-grams
    gt_ngrams = [tuple(gt_words[i:i+n]) for i in range(len(gt_words) - n + 1)]
    ex_ngrams = [tuple(ex_words[i:i+n]) for i in range(len(ex_words) - n + 1)]

    # Calculate overlap
    gt_set = set(gt_ngrams)
    ex_set = set(ex_ngrams)
    overlap = len(gt_set & ex_set)

    # Return as ratio of ground truth n-grams
    return overlap / len(gt_ngrams)


def _simple_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute edit distance for word sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def _longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


@dataclass
class OCREvaluationResult:
    """Results for evaluating a single document."""

    document_id: str
    image_path: str
    ground_truth_text: str
    extracted_text: str

    # Set-based metrics (position-agnostic)
    precision: float
    recall: float
    f1_score: float
    character_accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    word_count_gt: int
    word_count_ocr: int

    # Order-aware metrics (sequence quality)
    word_error_rate: float
    sequence_accuracy: float
    lcs_ratio: float
    normalized_edit_distance: float

    # N-gram overlap (local ordering)
    bigram_overlap: float
    trigram_overlap: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def print_summary(self):
        """Print a summary of this result."""
        print(f"Document: {self.document_id[:16]}...")
        print(f"  Path: {self.image_path}")
        print(f"  Words: GT={self.word_count_gt}, OCR={self.word_count_ocr}")
        print()
        print("  Set-based Metrics (position-agnostic):")
        print(f"    Precision: {self.precision:.3f}")
        print(f"    Recall: {self.recall:.3f}")
        print(f"    F1 Score: {self.f1_score:.3f}")
        print()
        print("  Order-aware Metrics (sequence quality):")
        print(f"    Word Error Rate: {self.word_error_rate:.3f}")
        print(f"    Sequence Accuracy: {self.sequence_accuracy:.3f}")
        print(f"    LCS Ratio: {self.lcs_ratio:.3f}")
        print()
        print("  N-gram Overlap (local ordering):")
        print(f"    Bigram Overlap: {self.bigram_overlap:.3f}")
        print(f"    Trigram Overlap: {self.trigram_overlap:.3f}")


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results across multiple documents."""

    evaluation_timestamp: str
    corpus_path: str
    pipeline_configuration: Dict[str, Any]
    total_documents: int

    # Set-based metrics (position-agnostic)
    avg_precision: float
    avg_recall: float
    avg_f1: float
    median_precision: float
    median_recall: float
    median_f1: float
    min_f1: float
    max_f1: float

    # Order-aware metrics
    avg_word_error_rate: float
    avg_sequence_accuracy: float
    avg_lcs_ratio: float
    median_word_error_rate: float
    median_sequence_accuracy: float
    median_lcs_ratio: float

    # N-gram metrics
    avg_bigram_overlap: float
    avg_trigram_overlap: float

    processing_time_seconds: float
    per_document_results: List[Dict[str, Any]]

    def to_json(self, path: Path):
        """Export report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"✓ JSON report saved to: {path}")

    def to_csv(self, path: Path):
        """Export per-document results as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.per_document_results:
            print("No results to export")
            return

        # Get all keys from first result
        fieldnames = [
            'document_id', 'image_path',
            'word_count_gt', 'word_count_ocr',
            'precision', 'recall', 'f1_score', 'character_accuracy',
            'word_error_rate', 'sequence_accuracy', 'lcs_ratio',
            'bigram_overlap', 'trigram_overlap',
            'true_positives', 'false_positives', 'false_negatives'
        ]

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.per_document_results)

        print(f"✓ CSV report saved to: {path}")

    def print_summary(self):
        """Print console summary."""
        print("=" * 70)
        print("BENCHMARK REPORT")
        print("=" * 70)
        print(f"Timestamp: {self.evaluation_timestamp}")
        print(f"Corpus: {self.corpus_path}")
        print(f"Documents: {self.total_documents}")
        print(f"Processing Time: {self.processing_time_seconds:.2f}s")
        print()
        print("Set-based Metrics (position-agnostic):")
        print(f"  Average Precision: {self.avg_precision:.3f}")
        print(f"  Average Recall:    {self.avg_recall:.3f}")
        print(f"  Average F1 Score:  {self.avg_f1:.3f}")
        print()
        print(f"  Median Precision:  {self.median_precision:.3f}")
        print(f"  Median Recall:     {self.median_recall:.3f}")
        print(f"  Median F1 Score:   {self.median_f1:.3f}")
        print()
        print("Order-aware Metrics (sequence quality):")
        print(f"  Avg Word Error Rate:    {self.avg_word_error_rate:.3f} (lower is better)")
        print(f"  Avg Sequence Accuracy:  {self.avg_sequence_accuracy:.3f}")
        print(f"  Avg LCS Ratio:          {self.avg_lcs_ratio:.3f}")
        print()
        print(f"  Median Word Error Rate: {self.median_word_error_rate:.3f}")
        print(f"  Median Sequence Acc:    {self.median_sequence_accuracy:.3f}")
        print(f"  Median LCS Ratio:       {self.median_lcs_ratio:.3f}")
        print()
        print("N-gram Overlap (local ordering):")
        print(f"  Avg Bigram Overlap:  {self.avg_bigram_overlap:.3f}")
        print(f"  Avg Trigram Overlap: {self.avg_trigram_overlap:.3f}")
        print("=" * 70)


class OCRBenchmark:
    """
    Runs OCR evaluation across multiple documents.

    Evaluates extraction snapshots against ground truth data and generates
    comprehensive reports with per-document and aggregate metrics.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize benchmark with a corpus.

        Args:
            corpus: Corpus containing documents and ground truth
        """
        self.corpus = corpus

    def evaluate_extraction(
        self,
        snapshot_reference: str,
        ground_truth_dir: Optional[Path] = None,
        pipeline_config: Optional[Dict] = None
    ) -> BenchmarkReport:
        """
        Evaluate an extraction snapshot against ground truth.

        Args:
            snapshot_reference: Snapshot ID or reference
            ground_truth_dir: Directory containing ground truth files
                            (defaults to corpus/metadata/funsd_ground_truth)
            pipeline_config: Configuration used to create snapshot

        Returns:
            BenchmarkReport with detailed results
        """
        start_time = time.time()

        # Default ground truth directory
        if ground_truth_dir is None:
            ground_truth_dir = self.corpus.meta_dir / "funsd_ground_truth"

        if not ground_truth_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory not found: {ground_truth_dir}"
            )

        # Find snapshot directory
        snapshot_dir = self.corpus.extraction_snapshot_dir(
            extractor_id="pipeline", snapshot_id=snapshot_reference
        )

        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_dir}")

        # Find text files in snapshot
        text_dir = snapshot_dir / "text"
        if not text_dir.exists():
            raise FileNotFoundError(f"Text directory not found: {text_dir}")

        text_files = list(text_dir.glob("*.txt"))
        if not text_files:
            raise ValueError(f"No text files found in: {text_dir}")

        print(f"Evaluating {len(text_files)} documents...")

        # Evaluate each document
        results: List[OCREvaluationResult] = []
        for text_file in text_files:
            document_id = text_file.stem

            # Load extracted text
            extracted_text = text_file.read_text()

            # Load ground truth
            gt_file = ground_truth_dir / f"{document_id}.txt"
            if not gt_file.exists():
                print(f"⚠️  Ground truth not found for {document_id[:16]}..., skipping")
                continue

            ground_truth_text = gt_file.read_text()

            # Find image path
            # Look in corpus raw directory
            try:
                item = self.corpus.get_item(document_id)
                image_path = item.relpath
            except KeyError:
                image_path = "unknown"

            # Calculate metrics
            word_metrics = calculate_word_metrics(ground_truth_text, extracted_text)
            char_accuracy = calculate_character_accuracy(ground_truth_text, extracted_text)
            order_metrics = calculate_word_order_metrics(ground_truth_text, extracted_text)
            bigram_overlap = calculate_ngram_overlap(ground_truth_text, extracted_text, n=2)
            trigram_overlap = calculate_ngram_overlap(ground_truth_text, extracted_text, n=3)

            result = OCREvaluationResult(
                document_id=document_id,
                image_path=image_path,
                ground_truth_text=ground_truth_text,
                extracted_text=extracted_text,
                precision=word_metrics['precision'],
                recall=word_metrics['recall'],
                f1_score=word_metrics['f1_score'],
                character_accuracy=char_accuracy,
                true_positives=word_metrics['true_positives'],
                false_positives=word_metrics['false_positives'],
                false_negatives=word_metrics['false_negatives'],
                word_count_gt=word_metrics['word_count_gt'],
                word_count_ocr=word_metrics['word_count_ocr'],
                word_error_rate=order_metrics['word_error_rate'],
                sequence_accuracy=order_metrics['sequence_accuracy'],
                lcs_ratio=order_metrics['lcs_ratio'],
                normalized_edit_distance=order_metrics['normalized_edit_distance'],
                bigram_overlap=bigram_overlap,
                trigram_overlap=trigram_overlap,
            )

            results.append(result)

        if not results:
            raise ValueError("No documents were evaluated successfully")

        # Calculate aggregate metrics
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1_scores = [r.f1_score for r in results]
        wers = [r.word_error_rate for r in results]
        seq_accs = [r.sequence_accuracy for r in results]
        lcs_ratios = [r.lcs_ratio for r in results]
        bigrams = [r.bigram_overlap for r in results]
        trigrams = [r.trigram_overlap for r in results]

        processing_time = time.time() - start_time

        report = BenchmarkReport(
            evaluation_timestamp=datetime.now().isoformat(),
            corpus_path=str(self.corpus.root),
            pipeline_configuration=pipeline_config or {},
            total_documents=len(results),
            avg_precision=sum(precisions) / len(precisions),
            avg_recall=sum(recalls) / len(recalls),
            avg_f1=sum(f1_scores) / len(f1_scores),
            median_precision=sorted(precisions)[len(precisions) // 2],
            median_recall=sorted(recalls)[len(recalls) // 2],
            median_f1=sorted(f1_scores)[len(f1_scores) // 2],
            min_f1=min(f1_scores),
            max_f1=max(f1_scores),
            avg_word_error_rate=sum(wers) / len(wers),
            avg_sequence_accuracy=sum(seq_accs) / len(seq_accs),
            avg_lcs_ratio=sum(lcs_ratios) / len(lcs_ratios),
            median_word_error_rate=sorted(wers)[len(wers) // 2],
            median_sequence_accuracy=sorted(seq_accs)[len(seq_accs) // 2],
            median_lcs_ratio=sorted(lcs_ratios)[len(lcs_ratios) // 2],
            avg_bigram_overlap=sum(bigrams) / len(bigrams),
            avg_trigram_overlap=sum(trigrams) / len(trigrams),
            processing_time_seconds=processing_time,
            per_document_results=[r.to_dict() for r in results],
        )

        return report
