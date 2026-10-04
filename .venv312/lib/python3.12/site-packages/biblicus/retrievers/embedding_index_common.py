"""
Shared primitives for embedding-index retrievers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..chunking import ChunkerConfig, TextChunk, TokenizerConfig
from ..constants import RETRIEVAL_DIR_NAME
from ..corpus import Corpus
from ..embedding_providers import EmbeddingProviderConfig, _l2_normalize_rows
from ..frontmatter import parse_front_matter
from ..models import ExtractionSnapshotReference, parse_extraction_snapshot_reference


class ChunkRecord(BaseModel):
    """
    Minimal persisted representation of a chunk.

    :ivar item_id: Item identifier that produced the chunk.
    :vartype item_id: str
    :ivar span_start: Inclusive start character offset.
    :vartype span_start: int
    :ivar span_end: Exclusive end character offset.
    :vartype span_end: int
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "ChunkRecord":
        if self.span_end <= self.span_start:
            raise ValueError("chunk span_end must be greater than span_start")
        return self


class EmbeddingIndexConfiguration(BaseModel):
    """
    Configuration for embedding-index retrievers.

    :ivar extraction_snapshot: Optional extraction snapshot reference in the form extractor_id:snapshot_id.
    :vartype extraction_snapshot: str or None
    :ivar chunker: Chunker configuration.
    :vartype chunker: biblicus.chunking.ChunkerConfig
    :ivar tokenizer: Optional tokenizer configuration.
    :vartype tokenizer: biblicus.chunking.TokenizerConfig or None
    :ivar embedding_provider: Embedding provider configuration.
    :vartype embedding_provider: biblicus.embedding_providers.EmbeddingProviderConfig
    :ivar snippet_characters: Optional maximum character count for returned evidence text.
    :vartype snippet_characters: int or None
    :ivar maximum_cache_total_items: Optional maximum number of vectors cached per scan batch.
    :vartype maximum_cache_total_items: int or None
    :ivar maximum_cache_total_characters: Optional maximum characters cached per scan batch.
    :vartype maximum_cache_total_characters: int or None
    """

    model_config = ConfigDict(extra="forbid")

    snippet_characters: Optional[int] = Field(default=None, ge=1)
    maximum_cache_total_items: Optional[int] = Field(default=None, ge=1)
    maximum_cache_total_characters: Optional[int] = Field(default=None, ge=1)
    extraction_snapshot: Optional[str] = None
    chunker: ChunkerConfig = Field(default_factory=lambda: ChunkerConfig(chunker_id="paragraph"))
    tokenizer: Optional[TokenizerConfig] = None
    embedding_provider: EmbeddingProviderConfig


def _extract_span_text(text: Optional[str], span: Tuple[int, int]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    span_start, span_end = span
    if span_start < 0 or span_end <= span_start:
        return text
    return text[span_start:span_end]


def _build_snippet(
    text: Optional[str], span: Tuple[int, int], max_chars: Optional[int]
) -> Optional[str]:
    if not isinstance(text, str):
        return None
    if max_chars is None:
        return _extract_span_text(text, span)
    if max_chars <= 0:
        return ""
    span_start, span_end = span
    if span_start < 0 or span_end <= span_start:
        return text[:max_chars]
    half_window = max_chars // 2
    snippet_start = max(span_start - half_window, 0)
    snippet_end = min(span_end + half_window, len(text))
    return text[snippet_start:snippet_end]


def resolve_extraction_reference(
    corpus: Corpus, configuration: EmbeddingIndexConfiguration
) -> Optional[ExtractionSnapshotReference]:
    """
    Resolve an extraction snapshot reference from an embedding-index configuration.

    :param corpus: Corpus associated with the configuration.
    :type corpus: Corpus
    :param configuration: Parsed embedding-index configuration.
    :type configuration: EmbeddingIndexConfiguration
    :return: Parsed extraction reference or None.
    :rtype: ExtractionSnapshotReference or None
    :raises FileNotFoundError: If an extraction snapshot is referenced but not present.
    """
    if not configuration.extraction_snapshot:
        return None
    extraction_reference = parse_extraction_snapshot_reference(configuration.extraction_snapshot)
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=extraction_reference.extractor_id,
        snapshot_id=extraction_reference.snapshot_id,
    )
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Missing extraction snapshot: {extraction_reference.as_string()}")
    return extraction_reference


def _load_text_from_item(
    corpus: Corpus,
    *,
    item_id: str,
    relpath: str,
    media_type: str,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> Optional[str]:
    if extraction_reference:
        extracted_text = corpus.read_extracted_text(
            extractor_id=extraction_reference.extractor_id,
            snapshot_id=extraction_reference.snapshot_id,
            item_id=item_id,
        )
        if isinstance(extracted_text, str):
            return extracted_text

    if media_type == "text/markdown":
        raw = (corpus.root / relpath).read_text(encoding="utf-8")
        return parse_front_matter(raw).body
    if media_type.startswith("text/"):
        return (corpus.root / relpath).read_text(encoding="utf-8")
    return None


def iter_text_payloads(
    corpus: Corpus, *, extraction_reference: Optional[ExtractionSnapshotReference]
) -> Iterator[Tuple[object, str]]:
    """
    Yield catalog items and their text payloads.

    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param extraction_reference: Optional extraction reference.
    :type extraction_reference: ExtractionSnapshotReference or None
    :yield: (catalog_item, text) pairs.
    :rtype: Iterator[tuple[object, str]]
    """
    catalog = corpus.load_catalog()
    for catalog_item in catalog.items.values():
        item_id = str(getattr(catalog_item, "id", ""))
        relpath = str(getattr(catalog_item, "relpath", ""))
        media_type = str(getattr(catalog_item, "media_type", ""))
        if not item_id or not relpath or not media_type:
            continue
        text = _load_text_from_item(
            corpus,
            item_id=item_id,
            relpath=relpath,
            media_type=media_type,
            extraction_reference=extraction_reference,
        )
        if not isinstance(text, str) or not text.strip():
            continue
        yield catalog_item, text


def collect_chunks(
    corpus: Corpus, *, configuration: EmbeddingIndexConfiguration
) -> Tuple[List[TextChunk], int]:
    """
    Collect chunks from text payloads in a corpus.

    :param corpus: Corpus to chunk.
    :type corpus: Corpus
    :param configuration: Parsed embedding-index configuration.
    :type configuration: EmbeddingIndexConfiguration
    :return: (chunks, text_item_count)
    :rtype: tuple[list[TextChunk], int]
    """
    tokenizer = configuration.tokenizer.build_tokenizer() if configuration.tokenizer else None
    chunker = configuration.chunker.build_chunker(tokenizer=tokenizer)
    extraction_reference = resolve_extraction_reference(corpus, configuration)

    chunks: List[TextChunk] = []
    next_chunk_id = 0
    text_items = 0
    for catalog_item, text in iter_text_payloads(corpus, extraction_reference=extraction_reference):
        text_items += 1
        item_id = str(getattr(catalog_item, "id"))
        item_chunks = chunker.chunk_text(
            item_id=item_id, text=text, starting_chunk_id=next_chunk_id
        )
        if item_chunks:
            next_chunk_id = item_chunks[-1].chunk_id + 1
            chunks.extend(item_chunks)
    return chunks, text_items


def chunks_to_records(chunks: Iterable[TextChunk]) -> List[ChunkRecord]:
    """
    Convert chunk objects to persisted chunk records.

    :param chunks: Chunk list.
    :type chunks: Iterable[TextChunk]
    :return: Chunk record list.
    :rtype: list[ChunkRecord]
    """
    records: List[ChunkRecord] = []
    for chunk in chunks:
        records.append(
            ChunkRecord(
                item_id=chunk.item_id,
                span_start=chunk.span_start,
                span_end=chunk.span_end,
            )
        )
    return records


def write_chunks_jsonl(path: Path, records: Iterable[ChunkRecord]) -> None:
    """
    Write chunk records as newline-delimited JSON.

    :param path: Destination path.
    :type path: pathlib.Path
    :param records: Chunk records.
    :type records: Iterable[ChunkRecord]
    :return: None.
    :rtype: None
    """
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), separators=(",", ":")) + "\n")


def read_chunks_jsonl(path: Path) -> List[ChunkRecord]:
    """
    Read chunk records from a JSON Lines file.

    :param path: Source path.
    :type path: pathlib.Path
    :return: Chunk record list.
    :rtype: list[ChunkRecord]
    """
    records: List[ChunkRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(ChunkRecord.model_validate(json.loads(line)))
    return records


def write_embeddings(path: Path, embeddings: np.ndarray) -> None:
    """
    Write embeddings to disk.

    :param path: Destination path.
    :type path: pathlib.Path
    :param embeddings: Embedding matrix.
    :type embeddings: numpy.ndarray
    :return: None.
    :rtype: None
    """
    np.save(path, embeddings.astype(np.float32))


def read_embeddings(path: Path, *, mmap: bool) -> np.ndarray:
    """
    Read embeddings from disk.

    :param path: Source path.
    :type path: pathlib.Path
    :param mmap: Whether to memory-map the file.
    :type mmap: bool
    :return: Embedding matrix.
    :rtype: numpy.ndarray
    """
    mode = "r" if mmap else None
    return np.load(path, mmap_mode=mode)


def cosine_similarity_scores(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity scores for a query vector.

    The embedding matrix must already be L2-normalized.

    :param embeddings: Embedding matrix of shape (n, d).
    :type embeddings: numpy.ndarray
    :param query_vector: Query vector of shape (d,).
    :type query_vector: numpy.ndarray
    :return: Score vector of shape (n,).
    :rtype: numpy.ndarray
    """
    query_vector = query_vector.astype(np.float32).reshape(-1)
    query_vector = _l2_normalize_rows(query_vector.reshape(1, -1)).reshape(-1)
    return embeddings @ query_vector


def artifact_paths_for_snapshot(*, snapshot_id: str, retriever_id: str) -> Dict[str, str]:
    """
    Build deterministic artifact relative paths for an embedding index snapshot.

    :param snapshot_id: Snapshot identifier.
    :type snapshot_id: str
    :param retriever_id: Retriever identifier.
    :type retriever_id: str
    :return: Mapping with keys embeddings and chunks.
    :rtype: dict[str, str]
    """
    prefix = f"{snapshot_id}.{retriever_id}"
    base_dir = Path(RETRIEVAL_DIR_NAME) / retriever_id / snapshot_id
    embeddings_relpath = str(base_dir / f"{prefix}.embeddings.npy")
    chunks_relpath = str(base_dir / f"{prefix}.chunks.jsonl")
    return {"embeddings": embeddings_relpath, "chunks": chunks_relpath}
