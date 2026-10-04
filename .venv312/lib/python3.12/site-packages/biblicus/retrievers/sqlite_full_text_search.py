"""
SQLite full-text search version five retriever for Biblicus.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..constants import RETRIEVAL_DIR_NAME
from ..corpus import Corpus
from ..frontmatter import parse_front_matter
from ..models import (
    Evidence,
    ExtractionSnapshotReference,
    QueryBudget,
    RetrievalResult,
    RetrievalSnapshot,
    parse_extraction_snapshot_reference,
)
from ..retrieval import (
    apply_budget,
    create_configuration_manifest,
    create_snapshot_manifest,
    hash_text,
)
from ..time import utc_now_iso


class SqliteFullTextSearchConfiguration(BaseModel):
    """
    Configuration for the SQLite full-text search retriever.

    :ivar chunk_size: Maximum characters per chunk.
    :vartype chunk_size: int
    :ivar chunk_overlap: Overlap characters between chunks.
    :vartype chunk_overlap: int
    :ivar snippet_characters: Maximum characters to include in evidence snippets.
    :vartype snippet_characters: int
    :ivar bm25_k1: BM25 k1 tuning parameter.
    :vartype bm25_k1: float
    :ivar bm25_b: BM25 b tuning parameter.
    :vartype bm25_b: float
    :ivar ngram_min: Minimum n-gram size for lexical tuning.
    :vartype ngram_min: int
    :ivar ngram_max: Maximum n-gram size for lexical tuning.
    :vartype ngram_max: int
    :ivar stop_words: Optional stop word policy or list.
    :vartype stop_words: str or list[str] or None
    :ivar field_weight_title: Relative weight for title field matches.
    :vartype field_weight_title: float
    :ivar field_weight_body: Relative weight for body field matches.
    :vartype field_weight_body: float
    :ivar field_weight_tags: Relative weight for tag field matches.
    :vartype field_weight_tags: float
    :ivar rerank_enabled: Whether to apply reranking to retrieved candidates.
    :vartype rerank_enabled: bool
    :ivar rerank_model: Reranker model identifier for metadata.
    :vartype rerank_model: str or None
    :ivar rerank_top_k: Number of candidates to rerank.
    :vartype rerank_top_k: int
    :ivar extraction_snapshot: Optional extraction snapshot reference in the form extractor_id:snapshot_id.
    :vartype extraction_snapshot: str or None
    """

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    snippet_characters: int = Field(default=400, ge=1)
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=1, ge=1)
    stop_words: Optional[Union[str, List[str]]] = None
    field_weight_title: float = Field(default=1.0, ge=0)
    field_weight_body: float = Field(default=1.0, ge=0)
    field_weight_tags: float = Field(default=1.0, ge=0)
    rerank_enabled: bool = False
    rerank_model: Optional[str] = None
    rerank_top_k: int = Field(default=10, ge=1)
    extraction_snapshot: Optional[str] = None

    @field_validator("stop_words")
    @classmethod
    def _validate_stop_words(
        cls, value: Optional[Union[str, List[str]]]
    ) -> Optional[Union[str, List[str]]]:
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() == "english":
                return "english"
            raise ValueError("stop_words must be 'english' or a list of strings")
        if not value:
            raise ValueError("stop_words list must not be empty")
        if any(not isinstance(token, str) or not token.strip() for token in value):
            raise ValueError("stop_words list must contain non-empty strings")
        return value

    @model_validator(mode="after")
    def _validate_ngram_range(self) -> "SqliteFullTextSearchConfiguration":
        if self.ngram_min > self.ngram_max:
            raise ValueError("Invalid ngram range: ngram_min must be <= ngram_max")
        if self.rerank_enabled and not self.rerank_model:
            raise ValueError("Rerank enabled requires rerank_model")
        return self


_ENGLISH_STOP_WORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class SqliteFullTextSearchRetriever:
    """
    SQLite full-text search version five retriever for practical local retrieval.

    :ivar retriever_id: Retriever identifier.
    :vartype retriever_id: str
    """

    retriever_id = "sqlite-full-text-search"

    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Build a full-text search version five index for the corpus.

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: RetrievalSnapshot
        """
        parsed_config = SqliteFullTextSearchConfiguration.model_validate(configuration)
        catalog = corpus.load_catalog()
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
        db_relpath = str(
            Path(RETRIEVAL_DIR_NAME)
            / self.retriever_id
            / snapshot.snapshot_id
            / f"{snapshot.snapshot_id}.sqlite"
        )
        db_path = corpus.root / db_relpath
        db_path.parent.mkdir(parents=True, exist_ok=True)
        extraction_reference = _resolve_extraction_reference(corpus, parsed_config)
        stats = _build_full_text_search_index(
            db_path=db_path,
            corpus=corpus,
            items=catalog.items.values(),
            configuration=parsed_config,
            extraction_reference=extraction_reference,
        )
        snapshot = snapshot.model_copy(update={"snapshot_artifacts": [db_relpath], "stats": stats})
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
        Query the SQLite full-text search index for evidence.

        :param corpus: Corpus associated with the snapshot.
        :type corpus: Corpus
        :param snapshot: Snapshot manifest to use for querying.
        :type snapshot: RetrievalSnapshot
        :param query_text: Query text to execute.
        :type query_text: str
        :param budget: Evidence selection budget.
        :type budget: QueryBudget
        :return: Retrieval results containing evidence.
        :rtype: RetrievalResult
        """
        parsed_config = SqliteFullTextSearchConfiguration.model_validate(
            snapshot.configuration.configuration
        )
        query_tokens = _tokenize_query(query_text)
        stop_words = _resolve_stop_words(parsed_config.stop_words)
        filtered_tokens = _apply_stop_words(query_tokens, stop_words)
        if not filtered_tokens:
            return RetrievalResult(
                query_text=query_text,
                budget=budget,
                snapshot_id=snapshot.snapshot_id,
                configuration_id=snapshot.configuration.configuration_id,
                retriever_id=snapshot.configuration.retriever_id,
                generated_at=utc_now_iso(),
                evidence=[],
                stats={"candidates": 0, "returned": 0},
            )
        db_path = _resolve_snapshot_db_path(corpus, snapshot)
        candidates = _query_full_text_search_index(
            db_path=db_path,
            query_text=" ".join(filtered_tokens),
            limit=_candidate_limit(budget.max_total_items + budget.offset),
            snippet_characters=parsed_config.snippet_characters,
        )
        sorted_candidates = _rank_candidates(candidates)
        evidence = _apply_rerank_if_enabled(
            sorted_candidates,
            query_tokens=filtered_tokens,
            snapshot=snapshot,
            budget=budget,
            rerank_enabled=parsed_config.rerank_enabled,
            rerank_top_k=parsed_config.rerank_top_k,
        )
        stats: Dict[str, object] = {"candidates": len(sorted_candidates), "returned": len(evidence)}
        if parsed_config.rerank_enabled:
            stats["reranked_candidates"] = min(len(sorted_candidates), parsed_config.rerank_top_k)
        return RetrievalResult(
            query_text=query_text,
            budget=budget,
            snapshot_id=snapshot.snapshot_id,
            configuration_id=snapshot.configuration.configuration_id,
            retriever_id=snapshot.configuration.retriever_id,
            generated_at=utc_now_iso(),
            evidence=evidence,
            stats=stats,
        )


def _candidate_limit(max_total_items: int) -> int:
    """
    Expand a candidate limit beyond the requested evidence count.

    :param max_total_items: Requested evidence count.
    :type max_total_items: int
    :return: Candidate limit for retriever search.
    :rtype: int
    """
    return max_total_items * 5


def _tokenize_query(query_text: str) -> List[str]:
    """
    Tokenize a query string into lowercased terms.

    :param query_text: Raw query text.
    :type query_text: str
    :return: Token list.
    :rtype: list[str]
    """
    tokens = [token for token in query_text.lower().split() if token]
    stripped = [re.sub(r"^[\W_]+|[\W_]+$", "", t) for t in tokens]
    return [t for t in stripped if t]


def _resolve_stop_words(value: Optional[Union[str, List[str]]]) -> Set[str]:
    """
    Resolve stop words based on a configuration value.

    :param value: Stop word configuration.
    :type value: str or list[str] or None
    :return: Stop word set.
    :rtype: set[str]
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return set(_ENGLISH_STOP_WORDS)
    return {token.strip().lower() for token in value if token.strip()}


def _apply_stop_words(tokens: List[str], stop_words: Set[str]) -> List[str]:
    """
    Filter query tokens by a stop word list.

    :param tokens: Token list.
    :type tokens: list[str]
    :param stop_words: Stop word set.
    :type stop_words: set[str]
    :return: Filtered token list.
    :rtype: list[str]
    """
    if not stop_words:
        return tokens
    return [token for token in tokens if token not in stop_words]


def _rank_candidates(candidates: List[Evidence]) -> List[Evidence]:
    """
    Sort evidence candidates by descending score.

    :param candidates: Evidence list to sort.
    :type candidates: list[Evidence]
    :return: Sorted evidence list.
    :rtype: list[Evidence]
    """
    return sorted(
        candidates, key=lambda evidence_item: (-evidence_item.score, evidence_item.item_id)
    )


def _rerank_score(text: str, query_tokens: List[str]) -> float:
    """
    Compute a simple rerank score using token overlap.

    :param text: Candidate text.
    :type text: str
    :param query_tokens: Query tokens.
    :type query_tokens: list[str]
    :return: Rerank score.
    :rtype: float
    """
    lower_text = text.lower() if text else ""
    return float(sum(1 for token in query_tokens if token in lower_text))


def _apply_rerank_if_enabled(
    candidates: List[Evidence],
    *,
    query_tokens: List[str],
    snapshot: RetrievalSnapshot,
    budget: QueryBudget,
    rerank_enabled: bool,
    rerank_top_k: int,
) -> List[Evidence]:
    """
    Rerank candidates when enabled, otherwise apply the budget to ranked results.

    :param candidates: Ranked candidate evidence.
    :type candidates: list[Evidence]
    :param query_tokens: Query tokens used for reranking.
    :type query_tokens: list[str]
    :param snapshot: Retrieval snapshot to annotate evidence with.
    :type snapshot: RetrievalSnapshot
    :param budget: Evidence selection budget.
    :type budget: QueryBudget
    :param rerank_enabled: Whether reranking is enabled.
    :type rerank_enabled: bool
    :param rerank_top_k: Maximum candidates to rerank.
    :type rerank_top_k: int
    :return: Evidence list respecting budget.
    :rtype: list[Evidence]
    """
    if not rerank_enabled:
        ranked = [
            evidence_item.model_copy(
                update={
                    "rank": index,
                    "configuration_id": snapshot.configuration.configuration_id,
                    "snapshot_id": snapshot.snapshot_id,
                }
            )
            for index, evidence_item in enumerate(candidates, start=1)
        ]
        return apply_budget(ranked, budget)

    rerank_limit = min(len(candidates), rerank_top_k)
    rerank_candidates = candidates[:rerank_limit]
    reranked: List[Evidence] = []
    for evidence_item in rerank_candidates:
        rerank_score = _rerank_score(evidence_item.text or "", query_tokens)
        reranked.append(
            evidence_item.model_copy(
                update={
                    "score": rerank_score,
                    "stage": "rerank",
                    "stage_scores": {"retrieve": evidence_item.score, "rerank": rerank_score},
                }
            )
        )
    reranked_sorted = _rank_candidates(reranked)
    ranked = [
        evidence_item.model_copy(
            update={
                "rank": index,
                "configuration_id": snapshot.configuration.configuration_id,
                "snapshot_id": snapshot.snapshot_id,
            }
        )
        for index, evidence_item in enumerate(reranked_sorted, start=1)
    ]
    return apply_budget(ranked, budget)


def _resolve_snapshot_db_path(corpus: Corpus, snapshot: RetrievalSnapshot) -> Path:
    """
    Resolve the SQLite index path for a retrieval snapshot.

    :param corpus: Corpus containing snapshot artifacts.
    :type corpus: Corpus
    :param snapshot: Retrieval snapshot manifest.
    :type snapshot: RetrievalSnapshot
    :return: Path to the SQLite index file.
    :rtype: Path
    :raises FileNotFoundError: If the snapshot does not have artifact paths.
    """
    if not snapshot.snapshot_artifacts:
        raise FileNotFoundError("Snapshot has no artifact paths to query")
    return corpus.root / snapshot.snapshot_artifacts[0]


def _ensure_full_text_search_version_five(conn: sqlite3.Connection) -> None:
    """
    Verify SQLite full-text search version five support in the current runtime.

    :param conn: SQLite connection to test.
    :type conn: sqlite3.Connection
    :return: None.
    :rtype: None
    :raises RuntimeError: If full-text search version five support is unavailable.
    """
    try:
        cursor = conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_full_text_search USING fts5(content)"
        )
        cursor.close()
        conn.execute("DROP TABLE IF EXISTS chunks_full_text_search")
    except sqlite3.OperationalError as operational_error:
        raise RuntimeError(
            "SQLite full-text search version five is required but not available in this Python build"
        ) from operational_error


def _create_full_text_search_schema(conn: sqlite3.Connection) -> None:
    """
    Create the full-text search schema in a fresh SQLite database.

    :param conn: SQLite connection for schema creation.
    :type conn: sqlite3.Connection
    :return: None.
    :rtype: None
    """
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_full_text_search USING fts5(
            content,
            item_id UNINDEXED,
            source_uri UNINDEXED,
            media_type UNINDEXED,
            relpath UNINDEXED,
            title UNINDEXED,
            start_offset UNINDEXED,
            end_offset UNINDEXED
        )
        """
    )


def _build_full_text_search_index(
    *,
    db_path: Path,
    corpus: Corpus,
    items: Iterable[object],
    configuration: SqliteFullTextSearchConfiguration,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> Dict[str, int]:
    """
    Build a full-text search index from corpus items.

    :param db_path: Destination SQLite database path.
    :type db_path: Path
    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param items: Catalog items to index.
    :type items: Iterable[object]
    :param configuration: Chunking and snippet configuration.
    :type configuration: SqliteFullTextSearchConfiguration
    :return: Index statistics.
    :rtype: dict[str, int]
    """
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(str(db_path))
    try:
        _ensure_full_text_search_version_five(connection)
        _create_full_text_search_schema(connection)
        chunk_count = 0
        item_count = 0
        text_item_count = 0
        for catalog_item in items:
            item_count += 1
            media_type = getattr(catalog_item, "media_type", "")
            relpath = getattr(catalog_item, "relpath", "")
            item_text = _load_text_from_item(
                corpus,
                item_id=str(getattr(catalog_item, "id", "")),
                relpath=str(relpath),
                media_type=str(media_type),
                extraction_reference=extraction_reference,
            )
            if item_text is None:
                continue
            text_item_count += 1
            title = getattr(catalog_item, "title", None)
            for start_offset, end_offset, chunk in _iter_chunks(
                item_text,
                chunk_size=configuration.chunk_size,
                chunk_overlap=configuration.chunk_overlap,
            ):
                connection.execute(
                    """
                    INSERT INTO chunks_full_text_search (
                        content,
                        item_id,
                        source_uri,
                        media_type,
                        relpath,
                        title,
                        start_offset,
                        end_offset
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk,
                        str(getattr(catalog_item, "id")),
                        getattr(catalog_item, "source_uri", None),
                        str(media_type),
                        str(relpath),
                        str(title) if title is not None else None,
                        start_offset,
                        end_offset,
                    ),
                )
                chunk_count += 1
        connection.commit()
        return {
            "items": item_count,
            "text_items": text_item_count,
            "chunks": chunk_count,
            "bytes": db_path.stat().st_size if db_path.exists() else 0,
        }
    finally:
        connection.close()


def _load_text_from_item(
    corpus: Corpus,
    *,
    item_id: str,
    relpath: str,
    media_type: str,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> Optional[str]:
    """
    Load text content from a catalog item.

    :param corpus: Corpus containing the content.
    :type corpus: Corpus
    :param item_id: Item identifier.
    :type item_id: str
    :param relpath: Relative path to the content.
    :type relpath: str
    :param media_type: Media type for the content.
    :type media_type: str
    :param extraction_reference: Optional extraction snapshot reference.
    :type extraction_reference: ExtractionSnapshotReference or None
    :return: Text payload or None if not text.
    :rtype: str or None
    """
    if extraction_reference:
        extracted_text = corpus.read_extracted_text(
            extractor_id=extraction_reference.extractor_id,
            snapshot_id=extraction_reference.snapshot_id,
            item_id=item_id,
        )
        if isinstance(extracted_text, str) and extracted_text.strip():
            return extracted_text

    content_path = corpus.root / relpath
    raw_bytes = content_path.read_bytes()
    if media_type == "text/markdown":
        markdown_text = raw_bytes.decode("utf-8")
        parsed_document = parse_front_matter(markdown_text)
        return parsed_document.body
    if media_type.startswith("text/"):
        return raw_bytes.decode("utf-8")
    return None


def _resolve_extraction_reference(
    corpus: Corpus,
    configuration: SqliteFullTextSearchConfiguration,
) -> Optional[ExtractionSnapshotReference]:
    """
    Resolve an extraction snapshot reference from a configuration.

    :param corpus: Corpus associated with the configuration.
    :type corpus: Corpus
    :param configuration: Parsed retriever configuration.
    :type configuration: SqliteFullTextSearchConfiguration
    :return: Parsed extraction reference or None.
    :rtype: ExtractionSnapshotReference or None
    :raises FileNotFoundError: If an extraction snapshot is referenced but not present.
    """
    if not configuration.extraction_snapshot:
        return corpus.latest_extraction_snapshot_reference()
    extraction_reference = parse_extraction_snapshot_reference(configuration.extraction_snapshot)
    snapshot_dir = corpus.extraction_snapshot_dir(
        extractor_id=extraction_reference.extractor_id,
        snapshot_id=extraction_reference.snapshot_id,
    )
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"Missing extraction snapshot: {extraction_reference.as_string()}")
    return extraction_reference


def _iter_chunks(
    text: str, *, chunk_size: int, chunk_overlap: int
) -> Iterable[Tuple[int, int, str]]:
    """
    Yield overlapping chunks of text for indexing.

    :param text: Text to chunk.
    :type text: str
    :param chunk_size: Maximum chunk size.
    :type chunk_size: int
    :param chunk_overlap: Overlap between chunks.
    :type chunk_overlap: int
    :return: Iterable of (start, end, chunk) tuples.
    :rtype: Iterable[tuple[int, int, str]]
    :raises ValueError: If the overlap is greater than or equal to the chunk size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    start_offset = 0
    text_length = len(text)
    while start_offset < text_length:
        end_offset = min(text_length, start_offset + chunk_size)
        yield start_offset, end_offset, text[start_offset:end_offset]
        if end_offset == text_length:
            break
        start_offset = end_offset - chunk_overlap


def _query_full_text_search_index(
    db_path: Path,
    query_text: str,
    limit: int,
    snippet_characters: int,
) -> List[Evidence]:
    """
    Query the SQLite full-text search index for evidence candidates.

    :param db_path: SQLite database path.
    :type db_path: Path
    :param query_text: Query text to execute.
    :type query_text: str
    :param limit: Maximum number of candidates to return.
    :type limit: int
    :param snippet_characters: Snippet length budget.
    :type snippet_characters: int
    :return: Evidence candidates.
    :rtype: list[Evidence]
    """
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            """
            SELECT
                content,
                item_id,
                source_uri,
                media_type,
                start_offset,
                end_offset,
                bm25(chunks_full_text_search) AS score
            FROM chunks_full_text_search
            WHERE chunks_full_text_search MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query_text, limit),
        ).fetchall()
        evidence_items: List[Evidence] = []
        for (
            content,
            item_id,
            source_uri,
            media_type,
            start_offset,
            end_offset,
            score,
        ) in rows:
            snippet_text = content[:snippet_characters]
            evidence_items.append(
                Evidence(
                    item_id=str(item_id),
                    source_uri=str(source_uri) if source_uri is not None else None,
                    media_type=str(media_type),
                    score=float(-score),
                    rank=1,
                    text=snippet_text,
                    content_ref=None,
                    span_start=int(start_offset) if start_offset is not None else None,
                    span_end=int(end_offset) if end_offset is not None else None,
                    stage="full-text-search",
                    configuration_id="",
                    snapshot_id="",
                    hash=hash_text(snippet_text),
                )
            )
        return evidence_items
    finally:
        connection.close()
