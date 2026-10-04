"""
Speech-to-Text benchmarking and evaluation system.

Provides comprehensive tools for evaluating STT pipeline performance against
ground truth transcriptions with detailed per-audio and aggregate metrics.

Supports standard ASR metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Word-level precision, recall, F1
"""

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from biblicus import Corpus


def normalize_transcript(text: str) -> str:
    """
    Normalize transcript for comparison.

    - Lowercases
    - Removes punctuation
    - Normalizes whitespace
    - Standard ASR normalization

    Args:
        text: Raw transcript text

    Returns:
        Normalized transcript string
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculate Word Error Rate (WER) using edit distance.

    WER = (S + D + I) / N
    Where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in reference

    Args:
        reference: Ground truth transcript
        hypothesis: STT output transcript

    Returns:
        Dictionary with WER, edit operations, and counts
    """
    ref_words = normalize_transcript(reference).split()
    hyp_words = normalize_transcript(hypothesis).split()

    # Build edit distance matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    # Fill matrix with edit distances
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # Backtrack to count operations
    i, j = len(ref_words), len(hyp_words)
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i == 0:
            insertions += 1
            j -= 1
        elif j == 0:
            deletions += 1
            i -= 1
        elif ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        else:
            # Check which operation produced current cell value
            current_cost = d[i][j]
            sub_cost = d[i-1][j-1] + 1 if i > 0 and j > 0 else float('inf')
            ins_cost = d[i][j-1] + 1 if j > 0 else float('inf')

            if sub_cost == current_cost:
                substitutions += 1
                i -= 1
                j -= 1
            elif ins_cost == current_cost:
                insertions += 1
                j -= 1
            else:
                deletions += 1
                i -= 1

    total_errors = substitutions + deletions + insertions
    wer = total_errors / len(ref_words) if len(ref_words) > 0 else 0.0

    return {
        "wer": wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
    }


def calculate_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculate Character Error Rate (CER) using edit distance.

    CER = (S + D + I) / N
    Where operations are at character level.

    Args:
        reference: Ground truth transcript
        hypothesis: STT output transcript

    Returns:
        Dictionary with CER and character-level counts
    """
    ref_chars = normalize_transcript(reference).replace(' ', '')
    hyp_chars = normalize_transcript(hypothesis).replace(' ', '')

    # Calculate edit distance at character level
    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j-1] + 1,  # substitution
                            d[i][j-1] + 1,      # insertion
                            d[i-1][j] + 1)      # deletion

    total_errors = d[len(ref_chars)][len(hyp_chars)]
    cer = total_errors / len(ref_chars) if len(ref_chars) > 0 else 0.0

    return {
        "cer": cer,
        "character_errors": total_errors,
        "reference_chars": len(ref_chars),
        "hypothesis_chars": len(hyp_chars),
    }


def calculate_word_metrics(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    Calculate word-level precision, recall, and F1 score.

    Treats transcripts as bag-of-words after normalization.

    Args:
        reference: Ground truth transcript
        hypothesis: STT output transcript

    Returns:
        Dictionary with precision, recall, f1_score
    """
    ref_words = set(normalize_transcript(reference).split())
    hyp_words = set(normalize_transcript(hypothesis).split())

    true_positives = len(ref_words & hyp_words)
    false_positives = len(hyp_words - ref_words)
    false_negatives = len(ref_words - hyp_words)

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
    }


@dataclass
class STTEvaluationResult:
    """Results for a single audio file transcription evaluation."""

    audio_id: str
    audio_path: str
    reference_text: str
    hypothesis_text: str

    # WER metrics
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    total_errors: int
    reference_words: int
    hypothesis_words: int

    # CER metrics
    cer: float
    character_errors: int
    reference_chars: int
    hypothesis_chars: int

    # Word-level metrics
    precision: float
    recall: float
    f1_score: float

    # Duration info (optional)
    audio_duration_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None


@dataclass
class STTBenchmarkReport:
    """
    Aggregate benchmark results across multiple audio files.

    Includes per-audio results and aggregate statistics.
    """

    evaluation_timestamp: str
    corpus_path: str
    provider_name: str
    provider_configuration: Dict[str, Any]
    total_audio_files: int

    # Aggregate WER metrics
    avg_wer: float
    median_wer: float
    avg_substitutions: float
    avg_deletions: float
    avg_insertions: float

    # Aggregate CER metrics
    avg_cer: float
    median_cer: float

    # Aggregate word-level metrics
    avg_precision: float
    avg_recall: float
    avg_f1: float
    median_f1: float

    # Per-audio results
    per_audio_results: List[STTEvaluationResult]

    def print_summary(self):
        """Print human-readable summary of benchmark results."""
        print("\n" + "="*80)
        print(f"STT BENCHMARK RESULTS: {self.provider_name}")
        print("="*80)
        print(f"Corpus: {self.corpus_path}")
        print(f"Audio Files: {self.total_audio_files}")
        print(f"Timestamp: {self.evaluation_timestamp}")
        print()

        print("WORD ERROR RATE (WER):")
        print(f"  Average: {self.avg_wer:.3f}")
        print(f"  Median:  {self.median_wer:.3f}")
        print()

        print("CHARACTER ERROR RATE (CER):")
        print(f"  Average: {self.avg_cer:.3f}")
        print(f"  Median:  {self.median_cer:.3f}")
        print()

        print("WORD-LEVEL METRICS:")
        print(f"  Precision: {self.avg_precision:.3f}")
        print(f"  Recall:    {self.avg_recall:.3f}")
        print(f"  F1 Score:  {self.avg_f1:.3f} (median: {self.median_f1:.3f})")
        print()

        print("ERROR BREAKDOWN:")
        print(f"  Substitutions: {self.avg_substitutions:.1f}")
        print(f"  Deletions:     {self.avg_deletions:.1f}")
        print(f"  Insertions:    {self.avg_insertions:.1f}")
        print("="*80 + "\n")

    def to_json(self, output_path: Path):
        """Export results to JSON file."""
        data = asdict(self)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def to_csv(self, output_path: Path):
        """Export per-audio results to CSV file."""
        if not self.per_audio_results:
            return

        fieldnames = [
            'audio_id', 'audio_path', 'wer', 'cer', 'precision', 'recall', 'f1_score',
            'substitutions', 'deletions', 'insertions', 'reference_words', 'hypothesis_words'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.per_audio_results:
                writer.writerow({
                    'audio_id': result.audio_id,
                    'audio_path': result.audio_path,
                    'wer': result.wer,
                    'cer': result.cer,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'substitutions': result.substitutions,
                    'deletions': result.deletions,
                    'insertions': result.insertions,
                    'reference_words': result.reference_words,
                    'hypothesis_words': result.hypothesis_words,
                })


class STTBenchmark:
    """
    Main STT benchmarking interface.

    Evaluates STT extraction performance against ground truth transcriptions.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize STT benchmark.

        Args:
            corpus: Corpus containing audio files and ground truth transcriptions
        """
        self.corpus = corpus

    def evaluate_extraction(
        self,
        snapshot_reference: str,
        ground_truth_dir: Path,
        provider_config: Optional[Dict[str, Any]] = None,
    ) -> STTBenchmarkReport:
        """
        Evaluate an STT extraction snapshot against ground truth.

        Args:
            snapshot_reference: Extraction snapshot ID
            ground_truth_dir: Directory containing ground truth .txt files
            provider_config: Optional provider configuration metadata

        Returns:
            STTBenchmarkReport with aggregate and per-audio results
        """
        # Load extraction snapshot
        snapshot_id = snapshot_reference
        text_dir = (
            self.corpus.root / "extracted" / "pipeline" / snapshot_id / "text"
        )

        if not text_dir.exists():
            raise FileNotFoundError(f"Text directory not found: {text_dir}")

        # Collect all transcriptions
        results = []

        for text_file in sorted(text_dir.glob("*.txt")):
            audio_id = text_file.stem

            # Load STT output
            hypothesis_text = text_file.read_text(encoding='utf-8')

            # Load ground truth
            gt_file = ground_truth_dir / f"{audio_id}.txt"
            if not gt_file.exists():
                continue  # Skip if no ground truth

            reference_text = gt_file.read_text(encoding='utf-8')

            # Calculate metrics
            wer_metrics = calculate_wer(reference_text, hypothesis_text)
            cer_metrics = calculate_cer(reference_text, hypothesis_text)
            word_metrics = calculate_word_metrics(reference_text, hypothesis_text)

            result = STTEvaluationResult(
                audio_id=audio_id,
                audio_path=str(text_file),
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                wer=wer_metrics["wer"],
                substitutions=wer_metrics["substitutions"],
                deletions=wer_metrics["deletions"],
                insertions=wer_metrics["insertions"],
                total_errors=wer_metrics["total_errors"],
                reference_words=wer_metrics["reference_words"],
                hypothesis_words=wer_metrics["hypothesis_words"],
                cer=cer_metrics["cer"],
                character_errors=cer_metrics["character_errors"],
                reference_chars=cer_metrics["reference_chars"],
                hypothesis_chars=cer_metrics["hypothesis_chars"],
                precision=word_metrics["precision"],
                recall=word_metrics["recall"],
                f1_score=word_metrics["f1_score"],
            )

            results.append(result)

        if not results:
            raise ValueError("No audio files with ground truth found")

        # Calculate aggregate statistics
        wers = [r.wer for r in results]
        cers = [r.cer for r in results]
        f1s = [r.f1_score for r in results]

        def calculate_median(values):
            """Calculate median correctly for both even and odd length lists."""
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 1:
                return sorted_values[n // 2]
            else:
                return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0

        report = STTBenchmarkReport(
            evaluation_timestamp=datetime.utcnow().isoformat() + 'Z',
            corpus_path=str(self.corpus.root),
            provider_name=provider_config.get("provider", "unknown") if provider_config else "unknown",
            provider_configuration=provider_config or {},
            total_audio_files=len(results),
            avg_wer=sum(wers) / len(wers),
            median_wer=calculate_median(wers),
            avg_substitutions=sum(r.substitutions for r in results) / len(results),
            avg_deletions=sum(r.deletions for r in results) / len(results),
            avg_insertions=sum(r.insertions for r in results) / len(results),
            avg_cer=sum(cers) / len(cers),
            median_cer=calculate_median(cers),
            avg_precision=sum(r.precision for r in results) / len(results),
            avg_recall=sum(r.recall for r in results) / len(results),
            avg_f1=sum(f1s) / len(f1s),
            median_f1=calculate_median(f1s),
            per_audio_results=results,
        )

        return report
