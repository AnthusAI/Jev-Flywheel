"""
Graph extraction models for Biblicus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..constants import GRAPH_SCHEMA_VERSION


class GraphSchemaModel(BaseModel):
    """
    Base model for graph extraction schemas with strict validation.

    :ivar schema_version: Graph schema version.
    :vartype schema_version: int
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=GRAPH_SCHEMA_VERSION, ge=1)


class GraphConfigurationManifest(BaseModel):
    """
    Reproducible configuration for a graph extraction snapshot.

    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar extractor_id: Graph extractor identifier.
    :vartype extractor_id: str
    :ivar name: Human-readable configuration name.
    :vartype name: str
    :ivar created_at: International Organization for Standardization 8601 timestamp.
    :vartype created_at: str
    :ivar configuration: Extractor-specific configuration values.
    :vartype configuration: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    configuration_id: str
    extractor_id: str
    name: str
    created_at: str
    configuration: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(GraphSchemaModel):
    """
    Node record extracted from a corpus item.

    :ivar node_id: Deterministic node identifier.
    :vartype node_id: str
    :ivar node_type: Node type identifier.
    :vartype node_type: str
    :ivar label: Human-readable label.
    :vartype label: str
    :ivar properties: Node-specific properties.
    :vartype properties: dict[str, Any]
    """

    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(GraphSchemaModel):
    """
    Edge record extracted from a corpus item.

    :ivar edge_id: Deterministic edge identifier.
    :vartype edge_id: str
    :ivar src: Source node identifier.
    :vartype src: str
    :ivar dst: Destination node identifier.
    :vartype dst: str
    :ivar edge_type: Edge type identifier.
    :vartype edge_type: str
    :ivar weight: Edge weight.
    :vartype weight: float
    :ivar properties: Edge-specific properties.
    :vartype properties: dict[str, Any]
    """

    edge_id: str = Field(min_length=1)
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    edge_type: str = Field(min_length=1)
    weight: float = Field(default=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphExtractionResult(GraphSchemaModel):
    """
    Graph extraction output for a single item.

    :ivar item_id: Corpus item identifier.
    :vartype item_id: str
    :ivar nodes: Extracted graph nodes.
    :vartype nodes: list[GraphNode]
    :ivar edges: Extracted graph edges.
    :vartype edges: list[GraphEdge]
    :ivar metadata: Extractor metadata.
    :vartype metadata: dict[str, Any]
    """

    item_id: str = Field(min_length=1)
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphSnapshotManifest(BaseModel):
    """
    Immutable record describing a graph extraction snapshot.

    :ivar snapshot_id: Unique snapshot identifier.
    :vartype snapshot_id: str
    :ivar graph_id: Deterministic graph identifier.
    :vartype graph_id: str
    :ivar configuration: Configuration manifest for this snapshot.
    :vartype configuration: GraphConfigurationManifest
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar catalog_generated_at: Catalog timestamp used for the snapshot.
    :vartype catalog_generated_at: str
    :ivar extraction_snapshot: Extraction snapshot reference.
    :vartype extraction_snapshot: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for snapshot creation.
    :vartype created_at: str
    :ivar stats: Snapshot statistics.
    :vartype stats: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    graph_id: str
    configuration: GraphConfigurationManifest
    corpus_uri: str
    catalog_generated_at: str
    extraction_snapshot: str
    created_at: str
    stats: Dict[str, Any] = Field(default_factory=dict)


class GraphSnapshotReference(BaseModel):
    """
    Reference to a graph extraction snapshot.

    :ivar extractor_id: Graph extractor identifier.
    :vartype extractor_id: str
    :ivar snapshot_id: Graph snapshot identifier.
    :vartype snapshot_id: str
    """

    model_config = ConfigDict(extra="forbid")

    extractor_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)

    def as_string(self) -> str:
        """
        Serialize the reference as a single string.

        :return: Reference in the form extractor_id:snapshot_id.
        :rtype: str
        """
        return f"{self.extractor_id}:{self.snapshot_id}"


class GraphSnapshotListEntry(BaseModel):
    """
    Summary entry for a graph extraction snapshot stored in a corpus.

    :ivar extractor_id: Graph extractor identifier.
    :vartype extractor_id: str
    :ivar snapshot_id: Graph snapshot identifier.
    :vartype snapshot_id: str
    :ivar graph_id: Deterministic graph identifier.
    :vartype graph_id: str
    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar configuration_name: Human-readable configuration name.
    :vartype configuration_name: str
    :ivar catalog_generated_at: Catalog timestamp used for the snapshot.
    :vartype catalog_generated_at: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for snapshot creation.
    :vartype created_at: str
    :ivar stats: Snapshot statistics.
    :vartype stats: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    extractor_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    configuration_name: str = Field(min_length=1)
    catalog_generated_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    stats: Dict[str, object] = Field(default_factory=dict)


def parse_graph_snapshot_reference(value: str) -> GraphSnapshotReference:
    """
    Parse a graph snapshot reference in the form extractor_id:snapshot_id.

    :param value: Raw reference string.
    :type value: str
    :return: Parsed graph snapshot reference.
    :rtype: GraphSnapshotReference
    :raises ValueError: If the reference is not well formed.
    """
    if ":" not in value:
        raise ValueError("Graph snapshot reference must be extractor_id:snapshot_id")
    extractor_id, snapshot_id = value.split(":", 1)
    extractor_id = extractor_id.strip()
    snapshot_id = snapshot_id.strip()
    if not extractor_id or not snapshot_id:
        raise ValueError(
            "Graph snapshot reference must be extractor_id:snapshot_id with non-empty parts"
        )
    return GraphSnapshotReference(extractor_id=extractor_id, snapshot_id=snapshot_id)


class GraphExtractionItemSummary(BaseModel):
    """
    Summary record for a single item in a graph extraction snapshot.

    :ivar item_id: Corpus item identifier.
    :vartype item_id: str
    :ivar node_count: Number of nodes written for the item.
    :vartype node_count: int
    :ivar edge_count: Number of edges written for the item.
    :vartype edge_count: int
    :ivar status: Result status.
    :vartype status: str
    :ivar error_message: Optional error message.
    :vartype error_message: str or None
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    status: str = Field(min_length=1)
    error_message: Optional[str] = None
