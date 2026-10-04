"""
Co-occurrence graph extractor for Biblicus.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ...corpus import Corpus
from ...models import CatalogItem
from ..base import GraphExtractor
from ..models import GraphEdge, GraphExtractionResult, GraphNode, GraphSchemaModel


class CooccurrenceGraphConfig(GraphSchemaModel):
    """
    Configuration for the co-occurrence graph extractor.

    :ivar window_size: Token window size for co-occurrence counting.
    :vartype window_size: int
    :ivar min_cooccurrence: Minimum count to emit an edge.
    :vartype min_cooccurrence: int
    """

    model_config = ConfigDict(extra="forbid")

    window_size: int = Field(default=4, ge=2)
    min_cooccurrence: int = Field(default=1, ge=1)


class CooccurrenceGraphExtractor(GraphExtractor):
    """
    Graph extractor that builds a co-occurrence graph from token windows.
    """

    extractor_id = "cooccurrence"

    def validate_config(self, config: Dict[str, object]) -> BaseModel:
        """
        Validate configuration for the co-occurrence extractor.

        :param config: Raw configuration mapping.
        :type config: dict[str, object]
        :return: Parsed configuration.
        :rtype: CooccurrenceGraphConfig
        """
        return CooccurrenceGraphConfig.model_validate(config)

    def extract_graph(
        self,
        *,
        corpus: Corpus,
        item: CatalogItem,
        extracted_text: str,
        config: BaseModel,
    ) -> GraphExtractionResult:
        """
        Extract graph nodes and edges for a single item.

        :param corpus: Corpus containing the item.
        :type corpus: Corpus
        :param item: Catalog item to extract from.
        :type item: CatalogItem
        :param extracted_text: Text to analyze.
        :type extracted_text: str
        :param config: Parsed configuration model.
        :type config: BaseModel
        :return: Graph extraction results.
        :rtype: GraphExtractionResult
        """
        _ = corpus
        parsed = config if isinstance(config, CooccurrenceGraphConfig) else None
        if parsed is None:
            parsed = CooccurrenceGraphConfig.model_validate(config)
        tokens = _tokenize(extracted_text)
        if not tokens:
            return GraphExtractionResult(item_id=item.id)
        windows = _windowed(tokens, parsed.window_size)
        edge_counts = _count_cooccurrences(windows)
        nodes = _build_nodes(edge_counts)
        edges = _build_edges(edge_counts, min_count=parsed.min_cooccurrence)
        return GraphExtractionResult(item_id=item.id, nodes=nodes, edges=edges)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token]


def _windowed(tokens: List[str], window_size: int) -> Iterable[Tuple[str, ...]]:
    if window_size <= 0:
        return []
    windows: List[Tuple[str, ...]] = []
    for index in range(len(tokens)):
        window = tokens[index : index + window_size]
        if len(window) < 2:
            continue
        windows.append(tuple(window))
    return windows


def _count_cooccurrences(windows: Iterable[Tuple[str, ...]]) -> Counter[Tuple[str, str]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for window in windows:
        unique_tokens = sorted(set(window))
        for left, right in combinations(unique_tokens, 2):
            counts[(left, right)] += 1
    return counts


def _build_nodes(counts: Counter[Tuple[str, str]]) -> List[GraphNode]:
    terms = sorted({token for pair in counts.keys() for token in pair})
    nodes: List[GraphNode] = []
    for term in terms:
        nodes.append(
            GraphNode(
                node_id=f"term:{term}",
                node_type="term",
                label=term,
                properties={},
            )
        )
    return nodes


def _build_edges(
    counts: Counter[Tuple[str, str]], *, min_count: int
) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    for (left, right), count in sorted(counts.items()):
        if count < min_count:
            continue
        src = f"term:{left}"
        dst = f"term:{right}"
        edge_id = f"{src}|cooccurs|{dst}"
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                src=src,
                dst=dst,
                edge_type="cooccurs",
                weight=float(count),
                properties={},
            )
        )
    return edges
