"""
Named entity recognition graph extractor for Biblicus.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ...corpus import Corpus
from ...models import CatalogItem
from ..base import GraphExtractor
from ..models import GraphEdge, GraphExtractionResult, GraphNode, GraphSchemaModel


class NerEntitiesGraphConfig(GraphSchemaModel):
    """
    Configuration for the NER entities graph extractor.

    :ivar model: Named entity recognition model identifier.
    :vartype model: str
    :ivar min_entity_length: Minimum length for entity labels.
    :vartype min_entity_length: int
    :ivar include_item_node: Whether to emit an item node and mentions edges.
    :vartype include_item_node: bool
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    min_entity_length: int = Field(default=3, ge=1)
    include_item_node: bool = Field(default=True)


class NerEntitiesGraphExtractor(GraphExtractor):
    """
    Graph extractor that emits named entity nodes and mentions edges.
    """

    extractor_id = "ner-entities"

    def validate_config(self, config: Dict[str, object]) -> BaseModel:
        """
        Validate configuration for NER entity extraction.

        :param config: Raw configuration mapping.
        :type config: dict[str, object]
        :return: Parsed configuration.
        :rtype: NerEntitiesGraphConfig
        """
        return NerEntitiesGraphConfig.model_validate(config)

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
        parsed = config if isinstance(config, NerEntitiesGraphConfig) else None
        if parsed is None:
            parsed = NerEntitiesGraphConfig.model_validate(config)

        entities = _extract_entities(
            extracted_text=extracted_text,
            model_name=parsed.model,
            min_length=parsed.min_entity_length,
        )
        entity_counts = Counter(entity for entity, _ in entities)
        entity_types = {entity: label for entity, label in entities}

        nodes = _build_entity_nodes(entity_counts, entity_types)
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

        return GraphExtractionResult(item_id=item.id, nodes=nodes, edges=edges)


def _extract_entities(
    *,
    extracted_text: str,
    model_name: str,
    min_length: int,
) -> List[Tuple[str, str]]:
    try:
        import spacy
    except ImportError as exc:
        raise ValueError(
            "NER graph extraction requires spaCy. Install it with pip install spacy."
        ) from exc
    nlp = spacy.load(model_name)
    doc = nlp(extracted_text)
    entities: List[Tuple[str, str]] = []
    for ent in getattr(doc, "ents", []):
        label = getattr(ent, "label_", "ENTITY")
        text = ent.text.strip()
        if len(text) < min_length:
            continue
        entities.append((text, label))
    return entities


def _canonicalize(label: str) -> str:
    lowered = label.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or lowered


def _build_entity_nodes(
    entity_counts: Counter[str], entity_types: Dict[str, str]
) -> List[GraphNode]:
    nodes: List[GraphNode] = []
    for label in sorted(entity_counts.keys()):
        canonical = _canonicalize(label)
        nodes.append(
            GraphNode(
                node_id=f"entity:{canonical}",
                node_type="entity",
                label=label,
                properties={
                    "canonical": canonical,
                    "entity_type": entity_types.get(label, "ENTITY"),
                },
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
