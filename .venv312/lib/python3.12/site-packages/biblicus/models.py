"""
Pydantic models for Biblicus domain concepts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import COLLECTION_SCHEMA_VERSION, SCHEMA_VERSION
from .hooks import HookSpec


class CorpusConfig(BaseModel):
    """
    Canonical on-disk config for a local Biblicus corpus.

    :ivar schema_version: Version of the corpus config schema.
    :vartype schema_version: int
    :ivar created_at: International Organization for Standardization 8601 timestamp for corpus creation.
    :vartype created_at: str
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar raw_dir: Relative path to the raw items folder.
    :vartype raw_dir: str
    :ivar notes: Optional free-form notes for operators.
    :vartype notes: dict[str, Any] or None
    :ivar hooks: Optional hook specifications for corpus lifecycle events.
    :vartype hooks: list[HookSpec] or None
    :ivar collection: Optional collection membership metadata.
    :vartype collection: CollectionMembership or None
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    created_at: str
    corpus_uri: str
    raw_dir: str = "."
    notes: Optional[Dict[str, Any]] = None
    hooks: Optional[List[HookSpec]] = None
    source: Optional["RemoteCorpusSourceConfig"] = None
    collection: Optional["CollectionMembership"] = None

    @model_validator(mode="after")
    def _enforce_schema_version(self) -> "CorpusConfig":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported corpus config schema version: {self.schema_version}")
        return self


class IngestResult(BaseModel):
    """
    Minimal summary for an ingestion event.

    :ivar item_id: Universally unique identifier assigned to the ingested item.
    :vartype item_id: str
    :ivar relpath: Relative path to the raw item file.
    :vartype relpath: str
    :ivar sha256: Secure Hash Algorithm 256 digest of the stored bytes.
    :vartype sha256: str
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    relpath: str
    sha256: str


class RemoteCorpusSourceConfig(BaseModel):
    """
    Configuration for a remote corpus source.

    :ivar kind: Remote source kind (s3 or azure-blob).
    :vartype kind: str
    :ivar profile: Source profile name in user configuration.
    :vartype profile: str
    :ivar name: Optional local namespace for storage.
    :vartype name: str or None
    :ivar bucket: S3 bucket name.
    :vartype bucket: str or None
    :ivar container: Azure Blob container name.
    :vartype container: str or None
    :ivar prefix: Optional remote prefix to scope the mirror.
    :vartype prefix: str
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    name: Optional[str] = None
    bucket: Optional[str] = None
    container: Optional[str] = None
    prefix: str = Field(default="")

    @model_validator(mode="after")
    def _validate_source_kind(self) -> "RemoteCorpusSourceConfig":
        if self.kind not in {"s3", "azure-blob"}:
            raise ValueError(f"Unsupported remote source kind: {self.kind}")
        if self.kind == "s3":
            if not (isinstance(self.bucket, str) and self.bucket.strip()):
                raise ValueError("Remote S3 source requires bucket")
        if self.kind == "azure-blob":
            if not (isinstance(self.container, str) and self.container.strip()):
                raise ValueError("Remote Azure Blob source requires container")
        return self


class CollectionMembership(BaseModel):
    """
    Collection membership metadata for a corpus.

    :ivar collection_name: Collection name.
    :vartype collection_name: str
    :ivar corpus_name: Corpus name within the collection.
    :vartype corpus_name: str
    """

    model_config = ConfigDict(extra="forbid")

    collection_name: str = Field(min_length=1)
    corpus_name: str = Field(min_length=1)


class RemoteCorpusCollectionDiscovery(BaseModel):
    """
    Discovery configuration for a remote collection.

    :ivar mode: Discovery mode (subfolder or partition).
    :vartype mode: str
    :ivar depth: Subfolder depth to discover.
    :vartype depth: int
    :ivar include_root_files: Whether to include root files under a reserved corpus.
    :vartype include_root_files: bool
    """

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(min_length=1)
    depth: int = Field(default=1, ge=1)
    include_root_files: bool = False

    @model_validator(mode="after")
    def _validate_mode(self) -> "RemoteCorpusCollectionDiscovery":
        if self.mode not in {"subfolder", "partition"}:
            raise ValueError(f"Unsupported collection discovery mode: {self.mode}")
        return self


class RemoteCorpusCollectionConfig(BaseModel):
    """
    Configuration for a remote corpus collection.

    :ivar schema_version: Version of the collection config schema.
    :vartype schema_version: int
    :ivar created_at: International Organization for Standardization 8601 timestamp.
    :vartype created_at: str
    :ivar collection_name: Collection name.
    :vartype collection_name: str
    :ivar source: Remote source configuration.
    :vartype source: RemoteCorpusSourceConfig
    :ivar discovery: Discovery configuration.
    :vartype discovery: RemoteCorpusCollectionDiscovery
    :ivar corpus_root: Filesystem path to the corpus root directory.
    :vartype corpus_root: str
    :ivar auto_create: Whether to auto-create discovered corpora.
    :vartype auto_create: bool
    :ivar deletion_policy: Policy for missing remote folders (archive or delete).
    :vartype deletion_policy: str
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    created_at: str
    collection_name: str = Field(min_length=1)
    source: RemoteCorpusSourceConfig
    discovery: RemoteCorpusCollectionDiscovery
    corpus_root: str = Field(min_length=1)
    auto_create: bool = True
    deletion_policy: str = Field(default="archive", min_length=1)

    @model_validator(mode="after")
    def _validate_deletion_policy(self) -> "RemoteCorpusCollectionConfig":
        if self.schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported collection config schema version: {self.schema_version}")
        if self.deletion_policy not in {"archive", "delete"}:
            raise ValueError(f"Unsupported collection deletion policy: {self.deletion_policy}")
        return self


class RemoteCollectionPullResult(BaseModel):
    """
    Summary of a collection pull operation.

    :ivar discovered: Number of discovered subfolders or partitions.
    :vartype discovered: int
    :ivar created: Number of corpora created.
    :vartype created: int
    :ivar mirrored: Number of corpora mirrored.
    :vartype mirrored: int
    :ivar archived: Number of corpora archived.
    :vartype archived: int
    :ivar errored: Number of errors.
    :vartype errored: int
    """

    model_config = ConfigDict(extra="forbid")

    discovered: int = Field(default=0, ge=0)
    created: int = Field(default=0, ge=0)
    mirrored: int = Field(default=0, ge=0)
    archived: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)


class PipelineCorpusSelector(BaseModel):
    """
    Corpus selection for a pipeline recipe.

    :ivar path: Optional corpus path.
    :vartype path: str or None
    :ivar collection: Optional collection name or path.
    :vartype collection: str or None
    :ivar selector: Optional selector pattern for collection corpora.
    :vartype selector: str or None
    """

    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    collection: Optional[str] = None
    selector: Optional[str] = None

    @model_validator(mode="after")
    def _validate_selector(self) -> "PipelineCorpusSelector":
        has_path = isinstance(self.path, str) and self.path.strip()
        has_collection = isinstance(self.collection, str) and self.collection.strip()
        if has_path and has_collection:
            raise ValueError("Pipeline recipe must specify corpus path or collection, not both")
        if not has_path and not has_collection:
            raise ValueError("Pipeline recipe must specify corpus path or collection")
        if has_collection and not (self.selector and self.selector.strip()):
            raise ValueError("Pipeline recipe collection requires a selector")
        return self


class PipelineMirrorConfig(BaseModel):
    """
    Mirror configuration for a pipeline recipe.

    :ivar collection: Collection path or name to mirror before running.
    :vartype collection: str
    """

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(min_length=1)


class PipelineExtractionConfig(BaseModel):
    """
    Extraction configuration for a pipeline recipe.

    :ivar recipe: Path to extraction recipe YAML.
    :vartype recipe: str
    """

    model_config = ConfigDict(extra="forbid")

    recipe: str = Field(min_length=1)


class PipelineRetrievalConfig(BaseModel):
    """
    Retrieval configuration for a pipeline recipe.

    :ivar retriever: Retriever identifier.
    :vartype retriever: str
    :ivar configuration: Path to retriever configuration file.
    :vartype configuration: str
    """

    model_config = ConfigDict(extra="forbid")

    retriever: str = Field(min_length=1)
    configuration: str = Field(min_length=1)


class PipelineAnalysisConfig(BaseModel):
    """
    Analysis configuration for a pipeline recipe.

    :ivar kind: Analysis kind identifier.
    :vartype kind: str
    :ivar configuration: Path to analysis configuration file.
    :vartype configuration: str
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    configuration: str = Field(min_length=1)


class PipelineRecipeConfig(BaseModel):
    """
    Pipeline recipe configuration.

    :ivar corpus: Corpus selection information.
    :vartype corpus: PipelineCorpusSelector
    :ivar mirror: Optional mirror configuration.
    :vartype mirror: PipelineMirrorConfig or None
    :ivar extraction: Optional extraction configuration.
    :vartype extraction: PipelineExtractionConfig or None
    :ivar retrieval: Optional retrieval configuration.
    :vartype retrieval: PipelineRetrievalConfig or None
    :ivar analysis: Optional analysis configuration list.
    :vartype analysis: list[PipelineAnalysisConfig] or None
    """

    model_config = ConfigDict(extra="forbid")

    corpus: PipelineCorpusSelector
    mirror: Optional[PipelineMirrorConfig] = None
    extraction: Optional[PipelineExtractionConfig] = None
    retrieval: Optional[PipelineRetrievalConfig] = None
    analysis: Optional[List[PipelineAnalysisConfig]] = None


class RemoteSourceItem(BaseModel):
    """
    Remote source object metadata.

    :ivar key: Remote object key or blob name.
    :vartype key: str
    :ivar source_uri: Source uniform resource identifier.
    :vartype source_uri: str
    :ivar etag: Optional entity tag for change detection.
    :vartype etag: str or None
    :ivar last_modified: Optional International Organization for Standardization 8601 timestamp.
    :vartype last_modified: str or None
    :ivar size: Size of the object in bytes.
    :vartype size: int
    :ivar content_type: Optional media type.
    :vartype content_type: str or None
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    source_uri: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    size: int = Field(ge=0)
    content_type: Optional[str] = None


class RemoteSourcePullResult(BaseModel):
    """
    Summary of a remote source pull operation.

    :ivar listed: Number of remote items listed.
    :vartype listed: int
    :ivar downloaded: Number of new items downloaded.
    :vartype downloaded: int
    :ivar updated: Number of existing items updated.
    :vartype updated: int
    :ivar skipped: Number of items skipped (no change).
    :vartype skipped: int
    :ivar pruned: Number of local items pruned.
    :vartype pruned: int
    :ivar errored: Number of items that failed to process.
    :vartype errored: int
    """

    model_config = ConfigDict(extra="forbid")

    listed: int = Field(default=0, ge=0)
    downloaded: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    pruned: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)


class CatalogItem(BaseModel):
    """
    Catalog entry derived from a raw corpus item.

    :ivar id: Universally unique identifier of the item.
    :vartype id: str
    :ivar relpath: Relative path to the raw item file.
    :vartype relpath: str
    :ivar sha256: Secure Hash Algorithm 256 digest of the stored bytes.
    :vartype sha256: str
    :ivar bytes: Size of the raw item in bytes.
    :vartype bytes: int
    :ivar media_type: Internet Assigned Numbers Authority media type for the item.
    :vartype media_type: str
    :ivar title: Optional human title extracted from metadata.
    :vartype title: str or None
    :ivar tags: Tags extracted or supplied for the item.
    :vartype tags: list[str]
    :ivar metadata: Merged front matter or sidecar metadata.
    :vartype metadata: dict[str, Any]
    :ivar created_at: International Organization for Standardization 8601 timestamp when the item was first indexed.
    :vartype created_at: str
    :ivar source_uri: Optional source uniform resource identifier used at ingestion time.
    :vartype source_uri: str or None
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    relpath: str
    sha256: str
    bytes: int = Field(ge=0)
    media_type: str
    title: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    source_uri: Optional[str] = None


class CorpusCatalog(BaseModel):
    """
    Snapshot of the derived corpus catalog.

    :ivar schema_version: Version of the catalog schema.
    :vartype schema_version: int
    :ivar generated_at: International Organization for Standardization 8601 timestamp of catalog generation.
    :vartype generated_at: str
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar raw_dir: Relative path to the raw items folder.
    :vartype raw_dir: str
    :ivar latest_run_id: Latest extraction run identifier, if any.
    :vartype latest_run_id: str or None
    :ivar latest_snapshot_id: Latest retrieval snapshot identifier, if any.
    :vartype latest_snapshot_id: str or None
    :ivar items: Mapping of item IDs to catalog entries.
    :vartype items: dict[str, CatalogItem]
    :ivar order: Display order of item IDs (most recent first).
    :vartype order: list[str]
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    generated_at: str
    corpus_uri: str
    raw_dir: str = "."
    latest_run_id: Optional[str] = None
    latest_snapshot_id: Optional[str] = None
    items: Dict[str, CatalogItem] = Field(default_factory=dict)
    order: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_schema_version(self) -> "CorpusCatalog":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported catalog schema version: {self.schema_version}")
        return self


class ExtractionSnapshotReference(BaseModel):
    """
    Reference to an extraction snapshot.

    :ivar extractor_id: Extractor plugin identifier.
    :vartype extractor_id: str
    :ivar snapshot_id: Extraction snapshot identifier.
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


def parse_extraction_snapshot_reference(value: str) -> ExtractionSnapshotReference:
    """
    Parse an extraction snapshot reference in the form extractor_id:snapshot_id.

    :param value: Raw reference string.
    :type value: str
    :return: Parsed extraction snapshot reference.
    :rtype: ExtractionSnapshotReference
    :raises ValueError: If the reference is not well formed.
    """
    if ":" not in value:
        raise ValueError("Extraction snapshot reference must be extractor_id:snapshot_id")
    extractor_id, snapshot_id = value.split(":", 1)
    extractor_id = extractor_id.strip()
    snapshot_id = snapshot_id.strip()
    if not extractor_id or not snapshot_id:
        raise ValueError(
            "Extraction snapshot reference must be extractor_id:snapshot_id with non-empty parts"
        )
    return ExtractionSnapshotReference(extractor_id=extractor_id, snapshot_id=snapshot_id)


class ExtractionSnapshotListEntry(BaseModel):
    """
    Summary entry for an extraction snapshot stored in a corpus.

    :ivar extractor_id: Extractor plugin identifier.
    :vartype extractor_id: str
    :ivar snapshot_id: Extraction snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar configuration_name: Human-readable configuration name.
    :vartype configuration_name: str
    :ivar catalog_generated_at: Catalog timestamp used for the snapshot.
    :vartype catalog_generated_at: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for snapshot creation.
    :vartype created_at: str
    :ivar stats: Snapshot statistics.
    :vartype stats: dict[str, object]
    """

    model_config = ConfigDict(extra="forbid")

    extractor_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    configuration_name: str = Field(min_length=1)
    catalog_generated_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    stats: Dict[str, object] = Field(default_factory=dict)


class QueryBudget(BaseModel):
    """
    Evidence selection budget for retrieval.

    The budget constrains the *returned* evidence. It intentionally does not
    change how a backend scores candidates, only how many evidence items are
    selected and how much text is allowed through.

    :ivar max_total_items: Maximum number of evidence items to return.
    :vartype max_total_items: int
    :ivar offset: Number of ranked candidates to skip before selecting evidence.
        This enables simple pagination by re-running the same query with a
        higher offset.
    :vartype offset: int
    :ivar maximum_total_characters: Optional maximum total characters across evidence text.
    :vartype maximum_total_characters: int or None
    :ivar max_items_per_source: Optional cap per source uniform resource identifier.
    :vartype max_items_per_source: int or None
    """

    model_config = ConfigDict(extra="forbid")

    max_total_items: int = Field(ge=1)
    offset: int = Field(default=0, ge=0)
    maximum_total_characters: Optional[int] = Field(default=None, ge=1)
    max_items_per_source: Optional[int] = Field(default=None, ge=1)


class Evidence(BaseModel):
    """
    Structured retrieval evidence returned from a retriever.

    :ivar item_id: Item identifier that produced the evidence.
    :vartype item_id: str
    :ivar source_uri: Source uniform resource identifier from ingestion metadata.
    :vartype source_uri: str or None
    :ivar media_type: Media type for the evidence item.
    :vartype media_type: str
    :ivar score: Retrieval score (higher is better).
    :vartype score: float
    :ivar rank: Rank within the final evidence list (1-based).
    :vartype rank: int
    :ivar text: Optional text payload for the evidence.
    :vartype text: str or None
    :ivar content_ref: Optional reference for non-text content.
    :vartype content_ref: str or None
    :ivar span_start: Optional start offset in the source text.
    :vartype span_start: int or None
    :ivar span_end: Optional end offset in the source text.
    :vartype span_end: int or None
    :ivar stage: Retrieval stage label (for example, scan, full-text search, rerank).
    :vartype stage: str
    :ivar stage_scores: Optional per-stage scores for multi-stage retrieval.
    :vartype stage_scores: dict[str, float] or None
    :ivar configuration_id: Configuration identifier used to create the snapshot.
    :vartype configuration_id: str
    :ivar snapshot_id: Retrieval snapshot identifier.
    :vartype snapshot_id: str
    :ivar metadata: Optional metadata payload from the catalog item.
    :vartype metadata: dict[str, Any]
    :ivar hash: Optional content hash for provenance.
    :vartype hash: str or None
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    source_uri: Optional[str] = None
    media_type: str
    score: float
    rank: int = Field(ge=1)
    text: Optional[str] = None
    content_ref: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    stage: str
    stage_scores: Optional[Dict[str, float]] = None
    configuration_id: str
    snapshot_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hash: Optional[str] = None

    @model_validator(mode="after")
    def _require_text_or_reference(self) -> "Evidence":
        has_text = isinstance(self.text, str) and self.text.strip()
        has_ref = isinstance(self.content_ref, str) and self.content_ref.strip()
        if not has_text and not has_ref:
            raise ValueError("Evidence must include either text or content_ref")
        return self


class ConfigurationManifest(BaseModel):
    """
    Reproducible configuration for a retriever.

    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar retriever_id: Retriever identifier for the configuration.
    :vartype retriever_id: str
    :ivar name: Human-readable name for the configuration.
    :vartype name: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for configuration creation.
    :vartype created_at: str
    :ivar configuration: Retriever-specific configuration values.
    :vartype configuration: dict[str, Any]
    :ivar description: Optional human description.
    :vartype description: str or None
    """

    model_config = ConfigDict(extra="forbid")

    configuration_id: str
    retriever_id: str
    name: str
    created_at: str
    configuration: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class RetrievalSnapshot(BaseModel):
    """
    Immutable record of a retrieval snapshot.

    :ivar snapshot_id: Unique snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration: Configuration manifest for this snapshot.
    :vartype configuration: ConfigurationManifest
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar catalog_generated_at: Catalog timestamp used for the snapshot.
    :vartype catalog_generated_at: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for snapshot creation.
    :vartype created_at: str
    :ivar snapshot_artifacts: Relative paths to materialized artifacts.
    :vartype snapshot_artifacts: list[str]
    :ivar stats: Retriever-specific snapshot statistics.
    :vartype stats: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    configuration: ConfigurationManifest
    corpus_uri: str
    catalog_generated_at: str
    created_at: str
    snapshot_artifacts: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """
    Retrieval result bundle returned from a retriever query.

    :ivar query_text: Query text issued against the backend.
    :vartype query_text: str
    :ivar budget: Evidence selection budget applied to results.
    :vartype budget: QueryBudget
    :ivar snapshot_id: Retrieval snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration_id: Configuration identifier used for this query.
    :vartype configuration_id: str
    :ivar retriever_id: Retriever identifier used for this query.
    :vartype retriever_id: str
    :ivar generated_at: International Organization for Standardization 8601 timestamp for the query result.
    :vartype generated_at: str
    :ivar evidence: Evidence objects selected under the budget.
    :vartype evidence: list[Evidence]
    :ivar stats: Backend-specific query statistics.
    :vartype stats: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    query_text: str
    budget: QueryBudget
    snapshot_id: str
    configuration_id: str
    retriever_id: str
    generated_at: str
    evidence: List[Evidence] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class ExtractedText(BaseModel):
    """
    Text payload produced by an extractor plugin.

    :ivar text: Extracted text content.
    :vartype text: str
    :ivar producer_extractor_id: Extractor identifier that produced this text.
    :vartype producer_extractor_id: str
    :ivar source_stage_index: Optional pipeline stage index where this text originated.
    :vartype source_stage_index: int or None
    :ivar confidence: Optional confidence score from 0.0 to 1.0.
    :vartype confidence: float or None
    :ivar metadata: Optional structured metadata for passing data between pipeline stages.
    :vartype metadata: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    producer_extractor_id: str = Field(min_length=1)
    source_stage_index: Optional[int] = Field(default=None, ge=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionStageOutput(BaseModel):
    """
    In-memory representation of a pipeline stage output for a single item.

    :ivar stage_index: One-based pipeline stage index.
    :vartype stage_index: int
    :ivar extractor_id: Extractor identifier for the stage.
    :vartype extractor_id: str
    :ivar status: Stage status, extracted, skipped, or errored.
    :vartype status: str
    :ivar text: Extracted text content, when produced.
    :vartype text: str or None
    :ivar text_characters: Character count of the extracted text.
    :vartype text_characters: int
    :ivar producer_extractor_id: Extractor identifier that produced the text content.
    :vartype producer_extractor_id: str or None
    :ivar source_stage_index: Optional stage index that supplied the text for selection-style extractors.
    :vartype source_stage_index: int or None
    :ivar confidence: Optional confidence score from 0.0 to 1.0.
    :vartype confidence: float or None
    :ivar metadata: Optional structured metadata for passing data between pipeline stages.
    :vartype metadata: dict[str, Any]
    :ivar error_type: Optional error type name for errored stages.
    :vartype error_type: str or None
    :ivar error_message: Optional error message for errored stages.
    :vartype error_message: str or None
    """

    model_config = ConfigDict(extra="forbid")

    stage_index: int = Field(ge=1)
    extractor_id: str
    status: str
    text: Optional[str] = None
    text_characters: int = Field(default=0, ge=0)
    producer_extractor_id: Optional[str] = None
    source_stage_index: Optional[int] = Field(default=None, ge=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None


GraphExtractionResult = import_module("biblicus.graph.models").GraphExtractionResult
