"""
Specialized metrics for document understanding evaluation.

This module provides metrics beyond basic OCR accuracy, including:
- Entity extraction metrics (for receipts, forms)
- Layout detection metrics (for document structure)
"""

from biblicus.evaluation.metrics.entity_metrics import (
    EntityExtractionResult,
    calculate_entity_f1,
    calculate_entity_metrics,
    extract_entities_from_text,
    normalize_entity_value,
)

__all__ = [
    "EntityExtractionResult",
    "calculate_entity_f1",
    "calculate_entity_metrics",
    "extract_entities_from_text",
    "normalize_entity_value",
]
