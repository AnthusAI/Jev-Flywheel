"""
Entity extraction metrics for structured document understanding.

This module provides metrics for evaluating entity extraction from documents,
particularly useful for receipt OCR (SROIE dataset) where we need to evaluate
extraction of structured fields like company name, date, address, and total.

Metrics:
- Per-entity-type F1 score
- Exact match accuracy
- Fuzzy match accuracy (for partial matches)
- Overall entity extraction F1
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Common entity types for SROIE receipts
SROIE_ENTITY_TYPES = ["company", "date", "address", "total"]


@dataclass
class EntityExtractionResult:
    """Results for entity extraction evaluation on a single document."""

    document_id: str
    entity_type: str
    ground_truth: str
    extracted: str
    exact_match: bool
    fuzzy_match: bool
    similarity: float
    normalized_gt: str = ""
    normalized_extracted: str = ""


@dataclass
class EntityMetricsReport:
    """Aggregate entity extraction metrics across documents."""

    total_documents: int
    entity_types: List[str]
    per_type_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    per_document_results: List[EntityExtractionResult] = field(default_factory=list)


def normalize_entity_value(value: str, entity_type: str = "") -> str:
    """
    Normalize an entity value for comparison.

    Applies type-specific normalization:
    - Dates: Normalize separators, handle common formats
    - Totals: Extract numeric value, normalize currency symbols
    - General: Lowercase, strip whitespace, remove extra spaces

    :param value: Raw entity value.
    :type value: str
    :param entity_type: Type of entity for specific normalization.
    :type entity_type: str
    :return: Normalized entity value.
    :rtype: str
    """
    if not value:
        return ""

    # Basic normalization
    normalized = value.strip().lower()

    # Remove extra whitespace
    normalized = " ".join(normalized.split())

    if entity_type == "date":
        # Normalize date separators
        normalized = re.sub(r"[/\-.]", "-", normalized)
        # Remove common date prefixes
        normalized = re.sub(r"^(date:?\s*)", "", normalized)

    elif entity_type == "total":
        # Extract numeric value from total
        # Remove currency symbols and common prefixes
        normalized = re.sub(r"[$£€¥]", "", normalized)
        normalized = re.sub(r"^(total:?\s*|amount:?\s*|sum:?\s*)", "", normalized)
        # Keep only digits, decimal point, and comma
        match = re.search(r"[\d,]+\.?\d*", normalized)
        if match:
            normalized = match.group().replace(",", "")

    elif entity_type == "company":
        # Remove common company suffixes for matching
        suffixes = [r"\s+(inc\.?|ltd\.?|llc\.?|corp\.?|co\.?)$"]
        for suffix in suffixes:
            normalized = re.sub(suffix, "", normalized, flags=re.IGNORECASE)

    elif entity_type == "address":
        # Normalize address abbreviations
        abbreviations = {
            r"\bst\.?\b": "street",
            r"\brd\.?\b": "road",
            r"\bave\.?\b": "avenue",
            r"\bblvd\.?\b": "boulevard",
            r"\bdr\.?\b": "drive",
            r"\bapt\.?\b": "apartment",
            r"\bste\.?\b": "suite",
        }
        for pattern, replacement in abbreviations.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    return normalized


def calculate_string_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein distance.

    :param s1: First string.
    :type s1: str
    :param s2: Second string.
    :type s2: str
    :return: Similarity score between 0 and 1.
    :rtype: float
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Simple edit distance implementation
    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    # Use only two rows of the DP table
    previous_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    distance = previous_row[-1]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def calculate_entity_metrics(
    ground_truth: Dict[str, str],
    extracted: Dict[str, str],
    entity_types: Optional[List[str]] = None,
    fuzzy_threshold: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate entity extraction metrics comparing ground truth to extracted entities.

    :param ground_truth: Dictionary of ground truth entity values by type.
    :type ground_truth: Dict[str, str]
    :param extracted: Dictionary of extracted entity values by type.
    :type extracted: Dict[str, str]
    :param entity_types: List of entity types to evaluate (default: SROIE types).
    :type entity_types: List[str] or None
    :param fuzzy_threshold: Similarity threshold for fuzzy matching.
    :type fuzzy_threshold: float
    :return: Per-type and overall metrics.
    :rtype: Dict[str, Dict[str, float]]
    """
    if entity_types is None:
        entity_types = SROIE_ENTITY_TYPES

    results: Dict[str, Dict[str, float]] = {}

    total_exact = 0
    total_fuzzy = 0
    total_entities = 0

    for entity_type in entity_types:
        gt_value = ground_truth.get(entity_type, "")
        ext_value = extracted.get(entity_type, "")

        # Normalize values
        gt_normalized = normalize_entity_value(gt_value, entity_type)
        ext_normalized = normalize_entity_value(ext_value, entity_type)

        # Calculate similarity
        similarity = calculate_string_similarity(gt_normalized, ext_normalized)

        # Determine match type
        exact_match = gt_normalized == ext_normalized and gt_normalized != ""
        fuzzy_match = similarity >= fuzzy_threshold and gt_normalized != ""

        results[entity_type] = {
            "exact_match": 1.0 if exact_match else 0.0,
            "fuzzy_match": 1.0 if fuzzy_match else 0.0,
            "similarity": similarity,
            "has_ground_truth": 1.0 if gt_value else 0.0,
            "has_extraction": 1.0 if ext_value else 0.0,
        }

        if gt_value:  # Only count if ground truth exists
            total_entities += 1
            if exact_match:
                total_exact += 1
            if fuzzy_match:
                total_fuzzy += 1

    # Overall metrics
    results["overall"] = {
        "exact_accuracy": total_exact / total_entities if total_entities > 0 else 0.0,
        "fuzzy_accuracy": total_fuzzy / total_entities if total_entities > 0 else 0.0,
        "total_entities": float(total_entities),
        "exact_matches": float(total_exact),
        "fuzzy_matches": float(total_fuzzy),
    }

    return results


def calculate_entity_f1(
    ground_truth_entities: List[Dict[str, str]],
    extracted_entities: List[Dict[str, str]],
    entity_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate F1 scores for entity extraction across multiple documents.

    :param ground_truth_entities: List of ground truth entity dictionaries.
    :type ground_truth_entities: List[Dict[str, str]]
    :param extracted_entities: List of extracted entity dictionaries.
    :type extracted_entities: List[Dict[str, str]]
    :param entity_types: List of entity types to evaluate.
    :type entity_types: List[str] or None
    :return: Per-type and overall F1 scores.
    :rtype: Dict[str, Dict[str, float]]
    """
    if entity_types is None:
        entity_types = SROIE_ENTITY_TYPES

    # Count true positives, false positives, false negatives per type
    tp: Dict[str, int] = {et: 0 for et in entity_types}
    fp: Dict[str, int] = {et: 0 for et in entity_types}
    fn: Dict[str, int] = {et: 0 for et in entity_types}

    for gt, ext in zip(ground_truth_entities, extracted_entities):
        for entity_type in entity_types:
            gt_value = normalize_entity_value(gt.get(entity_type, ""), entity_type)
            ext_value = normalize_entity_value(ext.get(entity_type, ""), entity_type)

            if gt_value and ext_value:
                if gt_value == ext_value:
                    tp[entity_type] += 1
                else:
                    # Partial match counts as both FP and FN
                    fp[entity_type] += 1
                    fn[entity_type] += 1
            elif gt_value and not ext_value:
                fn[entity_type] += 1
            elif ext_value and not gt_value:
                fp[entity_type] += 1

    # Calculate F1 per type
    results: Dict[str, Dict[str, float]] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_type in entity_types:
        precision = tp[entity_type] / (tp[entity_type] + fp[entity_type]) if (tp[entity_type] + fp[entity_type]) > 0 else 0.0
        recall = tp[entity_type] / (tp[entity_type] + fn[entity_type]) if (tp[entity_type] + fn[entity_type]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": float(tp[entity_type]),
            "false_positives": float(fp[entity_type]),
            "false_negatives": float(fn[entity_type]),
        }

        total_tp += tp[entity_type]
        total_fp += fp[entity_type]
        total_fn += fn[entity_type]

    # Overall F1 (micro-averaged)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    results["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "true_positives": float(total_tp),
        "false_positives": float(total_fp),
        "false_negatives": float(total_fn),
    }

    return results


def extract_entities_from_text(
    text: str,
    entity_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Extract entities from OCR text using pattern matching.

    This is a simple extraction method for testing. Production systems
    would use more sophisticated NER or LLM-based extraction.

    :param text: OCR extracted text.
    :type text: str
    :param entity_patterns: Custom regex patterns per entity type.
    :type entity_patterns: Dict[str, str] or None
    :return: Extracted entity values.
    :rtype: Dict[str, str]
    """
    if entity_patterns is None:
        # Default patterns for SROIE receipts
        entity_patterns = {
            # Date patterns (various formats)
            "date": r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            # Total amount pattern (currency followed by number)
            "total": r"(?:total|amount|sum)[\s:]*[$£€]?\s*[\d,]+\.?\d*",
            # Company name is harder - often first line or after specific keywords
            "company": r"^[A-Z][A-Za-z\s&]+(?:Inc\.?|Ltd\.?|LLC|Corp\.?)?",
            # Address pattern (street number + name)
            "address": r"\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Drive|Dr)",
        }

    entities: Dict[str, str] = {}

    for entity_type, pattern in entity_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            entities[entity_type] = match.group().strip()
        else:
            entities[entity_type] = ""

    return entities
