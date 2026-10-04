"""
Embedding-index retriever that reads the embedding matrix via memory mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..corpus import Corpus
from ..models import (
    Evidence,
    ExtractionSnapshotReference,
    QueryBudget,
    RetrievalResult,
    RetrievalSnapshot,
)
from ..retrieval import (
    apply_budget,
    create_configuration_manifest,
    create_snapshot_manifest,
    hash_text,
)
from ..time import utc_now_iso
from .embedding_index_common import (
    ChunkRecord,
    EmbeddingIndexConfiguration,
    _build_snippet,
    _extract_span_text,
    artifact_paths_for_snapshot,
    chunks_to_records,
    collect_chunks,
    cosine_similarity_scores,
    read_chunks_jsonl,
    read_embeddings,
    resolve_extraction_reference,
    write_chunks_jsonl,
    write_embeddings,
)


class EmbeddingIndexFileRetriever:
    """
    Embedding retrieval retriever using memory-mapped similarity scanning.
    """

    retriever_id = "embedding-index-file"

    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Build an embedding index snapshot by chunking text payloads and materializing embeddings.

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: biblicus.models.RetrievalSnapshot
        """
        parsed_config = EmbeddingIndexConfiguration.model_validate(configuration)
        chunks, text_items = collect_chunks(corpus, configuration=parsed_config)

        provider = parsed_config.embedding_provider.build_provider()
        embeddings = provider.embed_texts([chunk.text for chunk in chunks]).astype(np.float32)

        configuration_manifest = create_configuration_manifest(
            retriever_id=self.retriever_id,
            name=configuration_name,
            configuration=parsed_config.model_dump(),
        )
        snapshot = create_snapshot_manifest(
            corpus,
            configuration=configuration_manifest,
            stats={},
            snapshot_artifacts=[],
        )

        paths = artifact_paths_for_snapshot(
            snapshot_id=snapshot.snapshot_id, retriever_id=self.retriever_id
        )
        embeddings_path = corpus.root / paths["embeddings"]
        chunks_path = corpus.root / paths["chunks"]
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        write_embeddings(embeddings_path, embeddings)
        write_chunks_jsonl(chunks_path, chunks_to_records(chunks))

        stats = {
            "items": len(corpus.load_catalog().items),
            "text_items": text_items,
            "chunks": len(chunks),
            "dimensions": (
                int(embeddings.shape[1])
                if embeddings.size
                else parsed_config.embedding_provider.dimensions
            ),
        }
        snapshot = snapshot.model_copy(
            update={
                "snapshot_artifacts": [paths["embeddings"], paths["chunks"]],
                "stats": stats,
            }
        )
        corpus.write_snapshot(snapshot)
        return snapshot

    def query(
        self,
        corpus: Corpus,
        *,
        snapshot: RetrievalSnapshot,
        query_text: str,
        budget: QueryBudget,
    ) -> RetrievalResult:
        """
        Query an embedding index snapshot and return ranked evidence.

        :param corpus: Corpus associated with the snapshot.
        :type corpus: Corpus
        :param snapshot: Snapshot manifest to use for querying.
        :type snapshot: biblicus.models.RetrievalSnapshot
        :param query_text: Query text to embed.
        :type query_text: str
        :param budget: Evidence selection budget.
        :type budget: biblicus.models.QueryBudget
        :return: Retrieval results containing evidence.
        :rtype: biblicus.models.RetrievalResult
        """
        parsed_config = EmbeddingIndexConfiguration.model_validate(
            snapshot.configuration.configuration
        )
        extraction_reference = resolve_extraction_reference(corpus, parsed_config)

        paths = artifact_paths_for_snapshot(
            snapshot_id=snapshot.snapshot_id, retriever_id=self.retriever_id
        )
        embeddings_path = corpus.root / paths["embeddings"]
        chunks_path = corpus.root / paths["chunks"]
        if not embeddings_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError("Embedding index artifacts are missing for this snapshot")

        embeddings = read_embeddings(embeddings_path, mmap=True).astype(np.float32)
        chunk_records = read_chunks_jsonl(chunks_path)
        if embeddings.shape[0] != len(chunk_records):
            raise ValueError(
                "Embedding index artifacts are inconsistent: "
                "embeddings row count does not match chunk record count"
            )

        provider = parsed_config.embedding_provider.build_provider()
        query_embedding = provider.embed_texts([query_text]).astype(np.float32)
        if query_embedding.shape[0] != 1:
            raise ValueError("Embedding provider returned an invalid query embedding shape")

        batch_rows = parsed_config.maximum_cache_total_items or 4096
        candidates = _top_indices_batched(
            embeddings=embeddings,
            query_vector=query_embedding[0],
            limit=_candidate_limit(budget.max_total_items + budget.offset),
            batch_rows=batch_rows,
        )
        evidence_items = _build_evidence(
            corpus,
            snapshot=snapshot,
            configuration=parsed_config,
            candidates=candidates,
            embeddings=embeddings,
            query_vector=query_embedding[0],
            chunk_records=chunk_records,
            extraction_reference=extraction_reference,
        )
        ranked = [
            item.model_copy(
                update={
                    "rank": index,
                    "configuration_id": snapshot.configuration.configuration_id,
                    "snapshot_id": snapshot.snapshot_id,
                }
            )
            for index, item in enumerate(evidence_items, start=1)
        ]
        evidence = apply_budget(ranked, budget)
        return RetrievalResult(
            query_text=query_text,
            budget=budget,
            snapshot_id=snapshot.snapshot_id,
            configuration_id=snapshot.configuration.configuration_id,
            retriever_id=snapshot.configuration.retriever_id,
            generated_at=utc_now_iso(),
            evidence=evidence,
            stats={"candidates": len(evidence_items), "returned": len(evidence)},
        )


def _candidate_limit(max_total_items: int, *, multiplier: int = 10) -> int:
    return max(1, int(max_total_items) * int(multiplier))


@dataclass(frozen=True)
class _ScoredIndex:
    score: float
    index: int


def _top_indices_batched(
    *, embeddings: np.ndarray, query_vector: np.ndarray, limit: int, batch_rows: int = 4096
) -> List[int]:
    if embeddings.size == 0:
        return []
    limit = min(int(limit), int(embeddings.shape[0]))

    best: List[_ScoredIndex] = []
    for start in range(0, embeddings.shape[0], int(batch_rows)):
        end = min(start + int(batch_rows), embeddings.shape[0])
        scores = cosine_similarity_scores(embeddings[start:end], query_vector)
        batch_limit = min(limit, int(scores.size))
        if batch_limit <= 0:
            continue
        indices = np.argpartition(-scores, batch_limit - 1)[:batch_limit]
        for local_index in indices:
            global_index = int(start + int(local_index))
            best.append(_ScoredIndex(score=float(scores[int(local_index)]), index=global_index))

    best.sort(key=lambda item: (-item.score, item.index))
    return [int(item.index) for item in best[:limit]]


def _build_evidence(
    corpus: Corpus,
    *,
    snapshot: RetrievalSnapshot,
    configuration: EmbeddingIndexConfiguration,
    candidates: List[int],
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    chunk_records: List[ChunkRecord],
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> List[Evidence]:
    catalog = corpus.load_catalog()
    evidence_items: List[Evidence] = []
    for idx in candidates:
        record = chunk_records[idx]
        catalog_item = catalog.items[record.item_id]
        text = _load_text_for_evidence(
            corpus,
            item_id=record.item_id,
            relpath=str(getattr(catalog_item, "relpath")),
            media_type=str(getattr(catalog_item, "media_type")),
            extraction_reference=extraction_reference,
        )
        span_text = _build_snippet(
            text, (record.span_start, record.span_end), configuration.snippet_characters
        )
        if span_text is None:
            span_text = _extract_span_text(text, (record.span_start, record.span_end))
        score = float(cosine_similarity_scores(embeddings[idx : idx + 1], query_vector)[0])
        evidence_items.append(
            Evidence(
                item_id=record.item_id,
                source_uri=getattr(catalog_item, "source_uri", None),
                media_type=str(getattr(catalog_item, "media_type")),
                score=score,
                rank=1,
                text=span_text,
                content_ref=None,
                span_start=record.span_start,
                span_end=record.span_end,
                stage=EmbeddingIndexFileRetriever.retriever_id,
                stage_scores=None,
                configuration_id=snapshot.configuration.configuration_id,
                snapshot_id=snapshot.snapshot_id,
                metadata=getattr(catalog_item, "metadata", {}) or {},
                hash=hash_text(span_text or ""),
            )
        )
    return evidence_items


def _load_text_for_evidence(
    corpus: Corpus,
    *,
    item_id: str,
    relpath: str,
    media_type: str,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> Optional[str]:
    from .embedding_index_common import _load_text_from_item

    return _load_text_from_item(
        corpus,
        item_id=item_id,
        relpath=relpath,
        media_type=media_type,
        extraction_reference=extraction_reference,
    )
