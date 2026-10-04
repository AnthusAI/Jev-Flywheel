"""
Deterministic term-frequency vector retriever.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator

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


class TfVectorConfiguration(BaseModel):
    """
    Configuration for the term-frequency vector retriever.

    :ivar extraction_snapshot: Optional extraction snapshot reference in the form extractor_id:snapshot_id.
    :vartype extraction_snapshot: str or None
    :ivar snippet_characters: Optional maximum character count for returned evidence text.
    :vartype snippet_characters: int or None
    :ivar pipeline: Optional pipeline configuration (index/query) passed through by orchestrators.
    :vartype pipeline: dict[str, Any] or None
    """

    model_config = ConfigDict(extra="forbid")

    extraction_snapshot: Optional[str] = None
    snippet_characters: Optional[int] = None
    pipeline: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _merge_index_pipeline(self) -> "TfVectorConfiguration":
        if not self.pipeline or not isinstance(self.pipeline, dict):
            return self
        index_config = self.pipeline.get("index")
        if not isinstance(index_config, dict):
            return self
        if self.extraction_snapshot is None:
            self.extraction_snapshot = index_config.get("extraction_snapshot")
        if self.snippet_characters is None:
            self.snippet_characters = index_config.get("snippet_characters")
        return self


class TfVectorRetriever:
    """
    Deterministic vector retriever using term-frequency cosine similarity.

    :ivar retriever_id: Retriever identifier.
    :vartype retriever_id: str
    """

    retriever_id = "tf-vector"

    def build_snapshot(
        self, corpus: Corpus, *, configuration_name: str, configuration: Dict[str, object]
    ) -> RetrievalSnapshot:
        """
        Register a vector retriever snapshot (no snapshot artifacts).

        :param corpus: Corpus to build against.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Retriever-specific configuration values.
        :type configuration: dict[str, object]
        :return: Snapshot manifest describing the build.
        :rtype: RetrievalSnapshot
        """
        parsed_config = TfVectorConfiguration.model_validate(configuration)
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
        Query the corpus using term-frequency cosine similarity.

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
        parsed_config = TfVectorConfiguration.model_validate(snapshot.configuration.configuration)
        query_tokens = _tokenize_text(query_text)
        if not query_tokens:
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
        query_vector = _term_frequencies(query_tokens)
        query_norm = _vector_norm(query_vector)
        catalog = corpus.load_catalog()
        extraction_reference = _resolve_extraction_reference(corpus, parsed_config)
        scored_candidates = _score_items(
            corpus,
            catalog.items.values(),
            query_tokens=query_tokens,
            query_vector=query_vector,
            query_norm=query_norm,
            extraction_reference=extraction_reference,
            snippet_characters=parsed_config.snippet_characters,
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
    corpus: Corpus, configuration: TfVectorConfiguration
) -> Optional[ExtractionSnapshotReference]:
    """
    Resolve an extraction snapshot reference from a configuration.

    :param corpus: Corpus associated with the configuration.
    :type corpus: Corpus
    :param configuration: Parsed vector configuration.
    :type configuration: TfVectorConfiguration
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
    corpus: Corpus, items: Iterable[object], configuration: TfVectorConfiguration
) -> int:
    """
    Count catalog items that represent text content.

    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param items: Catalog items to inspect.
    :type items: Iterable[object]
    :param configuration: Parsed vector configuration.
    :type configuration: TfVectorConfiguration
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


def _tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens.

    :param text: Input text.
    :type text: str
    :return: Token list.
    :rtype: list[str]
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_frequencies(tokens: List[str]) -> Dict[str, float]:
    """
    Build term frequency weights from tokens.

    :param tokens: Token list.
    :type tokens: list[str]
    :return: Term frequency mapping.
    :rtype: dict[str, float]
    """
    frequencies: Dict[str, float] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0.0) + 1.0
    return frequencies


def _vector_norm(vector: Dict[str, float]) -> float:
    """
    Compute the Euclidean norm of a term-frequency vector.

    :param vector: Term frequency mapping.
    :type vector: dict[str, float]
    :return: Vector norm.
    :rtype: float
    """
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine_similarity(
    left: Dict[str, float],
    *,
    left_norm: float,
    right: Dict[str, float],
    right_norm: float,
) -> float:
    """
    Compute cosine similarity between two term-frequency vectors.

    :param left: Left term-frequency vector.
    :type left: dict[str, float]
    :param left_norm: Precomputed left vector norm.
    :type left_norm: float
    :param right: Right term-frequency vector.
    :type right: dict[str, float]
    :param right_norm: Precomputed right vector norm.
    :type right_norm: float
    :return: Cosine similarity score.
    :rtype: float
    """
    dot = 0.0
    if len(left) < len(right):
        for token, value in left.items():
            dot += value * right.get(token, 0.0)
    else:
        for token, value in right.items():
            dot += value * left.get(token, 0.0)
    return dot / (left_norm * right_norm)


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


def _build_snippet(text: str, span: Optional[Tuple[int, int]], *, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return text
    if not text:
        return ""
    if max_chars <= 0:
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
    *,
    query_tokens: List[str],
    query_vector: Dict[str, float],
    query_norm: float,
    extraction_reference: Optional[ExtractionSnapshotReference],
    snippet_characters: Optional[int],
) -> List[Evidence]:
    """
    Score catalog items and return evidence candidates.

    :param corpus: Corpus containing the items.
    :type corpus: Corpus
    :param items: Catalog items to score.
    :type items: Iterable[object]
    :param query_tokens: Tokenized query text.
    :type query_tokens: list[str]
    :param query_vector: Query term-frequency vector.
    :type query_vector: dict[str, float]
    :param query_norm: Query vector norm.
    :type query_norm: float
    :param extraction_reference: Optional extraction snapshot reference.
    :type extraction_reference: ExtractionSnapshotReference or None
    :param snippet_characters: Optional maximum character count for returned evidence text.
    :type snippet_characters: int or None
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
        tokens = _tokenize_text(item_text)
        if not tokens:
            continue
        vector = _term_frequencies(tokens)
        similarity = _cosine_similarity(
            query_vector, left_norm=query_norm, right=vector, right_norm=_vector_norm(vector)
        )
        if similarity <= 0:
            continue
        span = _find_first_match(item_text, query_tokens)
        span_start = span[0] if span else None
        span_end = span[1] if span else None
        evidence_text = _build_snippet(item_text, span, max_chars=snippet_characters)
        evidence_items.append(
            Evidence(
                item_id=str(getattr(catalog_item, "id")),
                source_uri=getattr(catalog_item, "source_uri", None),
                media_type=str(media_type),
                score=float(similarity),
                rank=1,
                text=evidence_text,
                content_ref=None,
                span_start=span_start,
                span_end=span_end,
                stage="tf-vector",
                configuration_id="",
                snapshot_id="",
                metadata=getattr(catalog_item, "metadata", {}) or {},
                hash=hash_text(evidence_text or ""),
            )
        )
    return evidence_items
