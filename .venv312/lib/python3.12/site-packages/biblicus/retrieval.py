"""
Shared retrieval helpers for Biblicus retrievers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    ConfigurationManifest,
    Evidence,
    QueryBudget,
    RetrievalSnapshot,
)
from .time import utc_now_iso

Corpus = Any


def create_configuration_manifest(
    *,
    retriever_id: str,
    name: str,
    configuration: Dict[str, Any],
    description: Optional[str] = None,
) -> ConfigurationManifest:
    """
    Create a deterministic configuration manifest from a retriever configuration.

    :param retriever_id: Retriever identifier for the configuration.
    :type retriever_id: str
    :param name: Human-readable configuration name.
    :type name: str
    :param configuration: Retriever-specific configuration values.
    :type configuration: dict[str, Any]
    :param description: Optional configuration description.
    :type description: str or None
    :return: Deterministic configuration manifest.
    :rtype: ConfigurationManifest
    """
    config_json = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    configuration_seed = f"{retriever_id}:{config_json}"
    configuration_id = hashlib.sha256(configuration_seed.encode("utf-8")).hexdigest()
    return ConfigurationManifest(
        configuration_id=configuration_id,
        retriever_id=retriever_id,
        name=name,
        created_at=utc_now_iso(),
        configuration=configuration,
        description=description,
    )


def create_snapshot_manifest(
    corpus: "Corpus",
    *,
    configuration: ConfigurationManifest,
    stats: Dict[str, Any],
    snapshot_artifacts: Optional[List[str]] = None,
) -> RetrievalSnapshot:
    """
    Create a retrieval snapshot manifest tied to the current catalog snapshot.

    :param corpus: Corpus used to generate the snapshot.
    :type corpus: Corpus
    :param configuration: Configuration manifest for the snapshot.
    :type configuration: ConfigurationManifest
    :param stats: Retriever-specific snapshot statistics.
    :type stats: dict[str, Any]
    :param snapshot_artifacts: Optional relative paths to materialized artifacts.
    :type snapshot_artifacts: list[str] or None
    :return: Snapshot manifest.
    :rtype: RetrievalSnapshot
    """
    catalog = corpus.load_catalog()
    created_at = utc_now_iso()
    snapshot_id = hashlib.sha256(
        f"{configuration.configuration_id}:{created_at}".encode("utf-8")
    ).hexdigest()
    return RetrievalSnapshot(
        snapshot_id=snapshot_id,
        configuration=configuration,
        corpus_uri=catalog.corpus_uri,
        catalog_generated_at=catalog.generated_at,
        created_at=created_at,
        snapshot_artifacts=list(snapshot_artifacts or []),
        stats=stats,
    )


def hash_text(text: str) -> str:
    """
    Hash a text payload for provenance.

    :param text: Text to hash.
    :type text: str
    :return: Secure Hash Algorithm 256 hex digest.
    :rtype: str
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_budget(evidence: Iterable[Evidence], budget: QueryBudget) -> List[Evidence]:
    """
    Apply a query budget to a ranked evidence list.

    :param evidence: Ranked evidence iterable (highest score first).
    :type evidence: Iterable[Evidence]
    :param budget: Budget constraints to enforce.
    :type budget: QueryBudget
    :return: Evidence list respecting the budget.
    :rtype: list[Evidence]
    """
    selected_evidence: List[Evidence] = []
    source_counts: Dict[str, int] = {}
    total_characters = 0
    skipped = 0

    for candidate_evidence in evidence:
        if skipped < budget.offset:
            skipped += 1
            continue

        if len(selected_evidence) >= budget.max_total_items:
            break

        source_key = candidate_evidence.source_uri or candidate_evidence.item_id
        if budget.max_items_per_source is not None:
            if source_counts.get(source_key, 0) >= budget.max_items_per_source:
                continue

        text_character_count = len(candidate_evidence.text or "")
        if budget.maximum_total_characters is not None:
            if total_characters + text_character_count > budget.maximum_total_characters:
                continue

        selected_evidence.append(candidate_evidence)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        total_characters += text_character_count

    return [
        evidence_item.model_copy(update={"rank": index})
        for index, evidence_item in enumerate(selected_evidence, start=1)
    ]
