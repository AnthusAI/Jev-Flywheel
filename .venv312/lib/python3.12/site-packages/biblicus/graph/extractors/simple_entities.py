"""
Deterministic entity and relationship graph extractor for Biblicus.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List

from pydantic import BaseModel, ConfigDict, Field

from ...corpus import Corpus
from ...models import CatalogItem
from ..base import GraphExtractor
from ..models import GraphEdge, GraphExtractionResult, GraphNode, GraphSchemaModel


class SimpleEntityGraphConfig(GraphSchemaModel):
    """
    Configuration for the simple entity graph extractor.

    :ivar min_entity_length: Minimum length for entity labels.
    :vartype min_entity_length: int
    :ivar include_item_node: Whether to emit an item node and mentions edges.
    :vartype include_item_node: bool
    :ivar max_entity_words: Maximum words per entity span.
    :vartype max_entity_words: int
    """

    model_config = ConfigDict(extra="forbid")

    min_entity_length: int = Field(default=3, ge=1)
    include_item_node: bool = Field(default=True)
    max_entity_words: int = Field(default=4, ge=1)


class SimpleEntitiesGraphExtractor(GraphExtractor):
    """
    Graph extractor that emits entities and sentence-level co-occurrence edges.
    """

    extractor_id = "simple-entities"

    def validate_config(self, config: Dict[str, object]) -> BaseModel:
        """
        Validate configuration for simple entity extraction.

        :param config: Raw configuration mapping.
        :type config: dict[str, object]
        :return: Parsed configuration.
        :rtype: SimpleEntityGraphConfig
        """
        safe_config: Dict[str, object]
        if isinstance(config, BaseModel):
            safe_config = config.model_dump()
        else:
            safe_config = dict(config)
        safe_config.pop("model", None)
        return SimpleEntityGraphConfig.model_validate(safe_config)

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
        parsed = config if isinstance(config, SimpleEntityGraphConfig) else None
        if parsed is None:
            parsed = self.validate_config(config)
        sentences = _split_sentences(extracted_text)
        entities_by_sentence = [
            _extract_entities(sentence, parsed.max_entity_words, parsed.min_entity_length)
            for sentence in sentences
        ]
        entity_counts = Counter(entity for entities in entities_by_sentence for entity in entities)

        nodes = _build_entity_nodes(entity_counts)
        edges: List[GraphEdge] = []

        if parsed.include_item_node:
            item_node = GraphNode(
                node_id=f"item:{item.id}",
                node_type="item",
                label=item.title or item.relpath,
                properties={"item_id": item.id},
            )
            nodes.insert(0, item_node)
            edges.extend(_build_mentions_edges(item_node.node_id, entity_counts))

        edges.extend(_build_relation_edges(entities_by_sentence))
        return GraphExtractionResult(item_id=item.id, nodes=nodes, edges=edges)


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"[.!?]+", text)
    return [segment.strip() for segment in raw if segment.strip()]


def _extract_entities(text: str, max_words: int, min_length: int) -> List[str]:
    entities: List[str] = []
    word_pattern = r"[A-Z][a-z]+"
    phrase_pattern = rf"\b{word_pattern}(?:\s+{word_pattern}){{0,{max_words - 1}}}\b"
    for match in re.findall(phrase_pattern, text):
        if len(match) >= min_length:
            entities.append(match)
    for match in re.findall(r"\b[A-Z]{2,}\b", text):
        if len(match) >= min_length:
            entities.append(match)
    return list(dict.fromkeys(entities))


def _canonicalize(label: str) -> str:
    lowered = label.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or lowered


def _build_entity_nodes(entity_counts: Counter[str]) -> List[GraphNode]:
    nodes: List[GraphNode] = []
    for label in sorted(entity_counts.keys()):
        canonical = _canonicalize(label)
        nodes.append(
            GraphNode(
                node_id=f"entity:{canonical}",
                node_type="entity",
                label=label,
                properties={"canonical": canonical},
            )
        )
    return nodes


def _build_mentions_edges(item_node_id: str, entity_counts: Counter[str]) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    for label, count in sorted(entity_counts.items()):
        canonical = _canonicalize(label)
        entity_id = f"entity:{canonical}"
        edge_id = f"{item_node_id}|mentions|{entity_id}"
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                src=item_node_id,
                dst=entity_id,
                edge_type="mentions",
                weight=float(count),
                properties={},
            )
        )
    return edges


def _build_relation_edges(entities_by_sentence: Iterable[List[str]]) -> List[GraphEdge]:
    counts: Counter[tuple[str, str]] = Counter()
    for entities in entities_by_sentence:
        canonical_entities = sorted({_canonicalize(label) for label in entities})
        for left, right in combinations(canonical_entities, 2):
            counts[(left, right)] += 1

    edges: List[GraphEdge] = []
    for (left, right), count in sorted(counts.items()):
        src = f"entity:{left}"
        dst = f"entity:{right}"
        edge_id = f"{src}|related_to|{dst}"
        edges.append(
            GraphEdge(
                edge_id=edge_id,
                src=src,
                dst=dst,
                edge_type="related_to",
                weight=float(count),
                properties={},
            )
        )
    return edges
