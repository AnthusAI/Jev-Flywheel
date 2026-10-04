"""
Naive full-scan retriever.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

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


class ScanConfiguration(BaseModel):
    """
    Configuration for the naive scan retriever.

    :ivar snippet_characters: Maximum characters to include in evidence snippets.
    :vartype snippet_characters: int
    :ivar extraction_snapshot: Optional extraction snapshot reference in the form extractor_id:snapshot_id.
    :vartype extraction_snapshot: str or None
    """

    model_config = ConfigDict(extra="forbid")

    snippet_characters: int = Field(default=400, ge=1)
    extraction_snapshot: Optional[str] = None


class ScanRetriever:
    """
    Naive retriever that scans all text items at query time.

    :ivar retriever_id: Retriever identifier.
    :vartype retriever_id: str
    """

    retriever_id = "scan"

    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Register a scan retriever snapshot (no snapshot artifacts).

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: RetrievalSnapshot
        """
        parsed_config = ScanConfiguration.model_validate(configuration)
        catalog = corpus.load_catalog()
        configuration_manifest = create_configuration_manifest(
            retriever_id=self.retriever_id,
            name=configuration_name,
            configuration=parsed_config.model_dump(),
        )
        stats = {
            "items": len(catalog.items),
            "text_items": _count_text_items(corpus, catalog.items.values(), parsed_config),
        }
        snapshot = create_snapshot_manifest(
            corpus,
            configuration=configuration_manifest,
            stats=stats,
            snapshot_artifacts=[],
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
        Query the corpus with a full scan.

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
        parsed_config = ScanConfiguration.model_validate(snapshot.configuration.configuration)
        catalog = corpus.load_catalog()
        extraction_reference = _resolve_extraction_reference(corpus, parsed_config)
        query_tokens = _tokenize_query(query_text)
        scored_candidates = _score_items(
            corpus,
            catalog.items.values(),
            query_tokens,
            parsed_config.snippet_characters,
            extraction_reference=extraction_reference,
        )
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda evidence_item: (-evidence_item.score, evidence_item.item_id),
        )
        ranked = [
            evidence_item.model_copy(
                update={
                    "rank": index,
                    "configuration_id": snapshot.configuration.configuration_id,
                    "snapshot_id": snapshot.snapshot_id,
                }
            )
            for index, evidence_item in enumerate(sorted_candidates, start=1)
        ]
        evidence = apply_budget(ranked, budget)
        stats = {"candidates": len(sorted_candidates), "returned": len(evidence)}
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


def _resolve_extraction_reference(
    corpus: Corpus, configuration: ScanConfiguration
) -> Optional[ExtractionSnapshotReference]:
    """
    Resolve an extraction snapshot reference from a configuration.

    :param corpus: Corpus associated with the configuration.
    :type corpus: Corpus
    :param configuration: Parsed scan configuration.
    :type configuration: ScanConfiguration
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


def _count_text_items(
    corpus: Corpus, items: Iterable[object], configuration: ScanConfiguration
) -> int:
    """
    Count catalog items that represent text content.

    When an extraction snapshot is configured, extracted artifacts are treated as text.

    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param items: Catalog items to inspect.
    :type items: Iterable[object]
    :param configuration: Parsed scan configuration.
    :type configuration: ScanConfiguration
    :return: Number of text items.
    :rtype: int
    """
    text_item_count = 0
    extraction_reference = _resolve_extraction_reference(corpus, configuration)
    for catalog_item in items:
        item_id = str(getattr(catalog_item, "id", ""))
        if extraction_reference and item_id:
            extracted_text = corpus.read_extracted_text(
                extractor_id=extraction_reference.extractor_id,
                snapshot_id=extraction_reference.snapshot_id,
                item_id=item_id,
            )
            if isinstance(extracted_text, str) and extracted_text.strip():
                text_item_count += 1
                continue
        media_type = getattr(catalog_item, "media_type", "")
        if media_type == "text/markdown" or str(media_type).startswith("text/"):
            text_item_count += 1
    return text_item_count


def _tokenize_query(query_text: str) -> List[str]:
    """
    Tokenize a query string for naive text matching.

    :param query_text: Raw query text.
    :type query_text: str
    :return: Lowercased non-empty tokens.
    :rtype: list[str]
    """
    return [token for token in query_text.lower().split() if token]


def _load_text_from_item(
    corpus: Corpus,
    *,
    item_id: str,
    relpath: str,
    media_type: str,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> Optional[str]:
    """
    Load a text payload from a catalog item.

    :param corpus: Corpus containing the item.
    :type corpus: Corpus
    :param item_id: Item identifier.
    :type item_id: str
    :param relpath: Relative path to the stored content.
    :type relpath: str
    :param media_type: Media type for the stored content.
    :type media_type: str
    :param extraction_reference: Optional extraction snapshot reference.
    :type extraction_reference: ExtractionSnapshotReference or None
    :return: Text payload or None if not decodable as text.
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


def _find_first_match(text: str, tokens: List[str]) -> Optional[Tuple[int, int]]:
    """
    Locate the earliest token match span in a text payload.

    :param text: Text to scan.
    :type text: str
    :param tokens: Query tokens.
    :type tokens: list[str]
    :return: Start/end span for the earliest match, or None if no matches.
    :rtype: tuple[int, int] or None
    """
    lower_text = text.lower()
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    for token in tokens:
        if not token:
            continue
        token_start = lower_text.find(token)
        if token_start == -1:
            continue
        token_end = token_start + len(token)
        if best_start is None or token_start < best_start:
            best_start = token_start
            best_end = token_end
    if best_start is None or best_end is None:
        return None
    return best_start, best_end


def _build_snippet(text: str, span: Optional[Tuple[int, int]], *, max_chars: int) -> str:
    """
    Build a snippet around a match span, constrained by a character budget.

    :param text: Source text to slice.
    :type text: str
    :param span: Match span to center on.
    :type span: tuple[int, int] or None
    :param max_chars: Maximum snippet length.
    :type max_chars: int
    :return: Snippet text.
    :rtype: str
    """
    if not text:
        return ""
    if span is None:
        return text[:max_chars]
    span_start, span_end = span
    half_window = max_chars // 2
    snippet_start = max(span_start - half_window, 0)
    snippet_end = min(span_end + half_window, len(text))
    return text[snippet_start:snippet_end]


def _score_items(
    corpus: Corpus,
    items: Iterable[object],
    tokens: List[str],
    snippet_characters: int,
    *,
    extraction_reference: Optional[ExtractionSnapshotReference],
) -> List[Evidence]:
    """
    Score catalog items by token frequency and return evidence candidates.

    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param items: Catalog items to score.
    :type items: Iterable[object]
    :param tokens: Query tokens to count.
    :type tokens: list[str]
    :param snippet_characters: Snippet length budget.
    :type snippet_characters: int
    :return: Evidence candidates with provisional ranks.
    :rtype: list[Evidence]
    """
    evidence_items: List[Evidence] = []
    for catalog_item in items:
        media_type = getattr(catalog_item, "media_type", "")
        relpath = getattr(catalog_item, "relpath", "")
        item_id = str(getattr(catalog_item, "id", ""))
        item_text = _load_text_from_item(
            corpus,
            item_id=item_id,
            relpath=relpath,
            media_type=str(media_type),
            extraction_reference=extraction_reference,
        )
        if item_text is None:
            continue
        lower_text = item_text.lower()
        match_score = sum(lower_text.count(token) for token in tokens)
        if match_score <= 0:
            continue
        span = _find_first_match(item_text, tokens)
        snippet = _build_snippet(item_text, span, max_chars=snippet_characters)
        span_start = span[0] if span else None
        span_end = span[1] if span else None
        evidence_items.append(
            Evidence(
                item_id=str(getattr(catalog_item, "id")),
                source_uri=getattr(catalog_item, "source_uri", None),
                media_type=str(media_type),
                score=float(match_score),
                rank=1,
                text=snippet,
                content_ref=None,
                span_start=span_start,
                span_end=span_end,
                stage="scan",
                configuration_id="",
                snapshot_id="",
                metadata=getattr(catalog_item, "metadata", {}) or {},
                hash=hash_text(snippet),
            )
        )

    return evidence_items
