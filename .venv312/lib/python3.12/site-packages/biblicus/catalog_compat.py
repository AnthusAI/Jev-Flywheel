"""
Helpers for loading corpus catalogs that carry foreign top-level item fields.

Downstream tools may attach editorial extensions beside the canonical Biblicus
catalog item shape. Those keys are folded into ``metadata`` on read so strict
``CatalogItem`` validation succeeds without product-specific schema branches.
"""

from __future__ import annotations

from typing import Any, Mapping

from .models import CatalogItem, CorpusCatalog

CATALOG_ITEM_ROOT_FIELDS = frozenset(CatalogItem.model_fields.keys())


def sanitize_catalog_item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize one catalog item mapping to the canonical root field set.

    Unknown top-level keys are merged into ``metadata`` so callers can keep
    strict ``CatalogItem`` validation while preserving extension data.
    """
    if not isinstance(item, Mapping):
        raise TypeError("Catalog item payload must be a mapping.")
    sanitized: dict[str, Any] = {}
    metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
    for key, value in item.items():
        if key == "metadata":
            continue
        if key in CATALOG_ITEM_ROOT_FIELDS:
            sanitized[key] = value
            continue
        if key in metadata and isinstance(metadata[key], dict) and isinstance(value, dict):
            metadata[key] = {**metadata[key], **value}
        else:
            metadata[key] = value
    sanitized["metadata"] = metadata
    return sanitized


def sanitize_catalog_payload(catalog_data: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize a full catalog document before ``CorpusCatalog`` validation.
    """
    if not isinstance(catalog_data, Mapping):
        raise TypeError("Catalog payload must be a mapping.")
    sanitized = dict(catalog_data)
    raw_items = catalog_data.get("items")
    if not isinstance(raw_items, Mapping):
        return sanitized
    sanitized["items"] = {
        item_id: sanitize_catalog_item_payload(item)
        for item_id, item in raw_items.items()
        if isinstance(item, Mapping)
    }
    return sanitized


def load_corpus_catalog(catalog_data: Mapping[str, Any]) -> CorpusCatalog:
    """
    Parse catalog JSON after folding foreign item extensions into metadata.
    """
    return CorpusCatalog.model_validate(sanitize_catalog_payload(catalog_data))
