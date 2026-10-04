"""
Pydantic models for analysis pipelines.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from ..ai.models import EmbeddingsClientConfig, LlmClientConfig
from ..constants import ANALYSIS_SCHEMA_VERSION
from ..models import ExtractionSnapshotReference
from .schema import AnalysisSchemaModel


class AnalysisConfigurationManifest(AnalysisSchemaModel):
    """
    Reproducible configuration for an analysis pipeline.

    :ivar configuration_id: Deterministic configuration identifier.
    :vartype configuration_id: str
    :ivar analysis_id: Analysis backend identifier.
    :vartype analysis_id: str
    :ivar name: Human-readable configuration name.
    :vartype name: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for configuration creation.
    :vartype created_at: str
    :ivar config: Analysis-specific configuration values.
    :vartype config: dict[str, Any]
    :ivar description: Optional human description.
    :vartype description: str or None
    """

    configuration_id: str
    analysis_id: str
    name: str
    created_at: str
    config: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class AnalysisRunInput(AnalysisSchemaModel):
    """
    Inputs required to execute an analysis snapshot.

    :ivar extraction_snapshot: Extraction snapshot reference for analysis inputs.
    :vartype extraction_snapshot: biblicus.models.ExtractionSnapshotReference
    """

    extraction_snapshot: ExtractionSnapshotReference


class AnalysisRunManifest(AnalysisSchemaModel):
    """
    Immutable record of an analysis snapshot.

    :ivar snapshot_id: Unique snapshot identifier.
    :vartype snapshot_id: str
    :ivar configuration: Configuration manifest for this run.
    :vartype configuration: AnalysisConfigurationManifest
    :ivar corpus_uri: Canonical uniform resource identifier for the corpus root.
    :vartype corpus_uri: str
    :ivar catalog_generated_at: Catalog timestamp used for the run.
    :vartype catalog_generated_at: str
    :ivar created_at: International Organization for Standardization 8601 timestamp for run creation.
    :vartype created_at: str
    :ivar input: Inputs used for this analysis snapshot.
    :vartype input: AnalysisRunInput
    :ivar artifact_paths: Relative paths to materialized artifacts.
    :vartype artifact_paths: list[str]
    :ivar stats: Analysis-specific run statistics.
    :vartype stats: dict[str, Any]
    """

    snapshot_id: str
    configuration: AnalysisConfigurationManifest
    corpus_uri: str
    catalog_generated_at: str
    created_at: str
    input: AnalysisRunInput
    artifact_paths: List[str] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


class ProfilingConfiguration(AnalysisSchemaModel):
    """
    Configuration for profiling analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar sample_size: Optional sample size for distribution metrics.
    :vartype sample_size: int or None
    :ivar min_text_characters: Optional minimum character count for extracted text inclusion.
    :vartype min_text_characters: int or None
    :ivar percentiles: Percentiles to compute for distributions.
    :vartype percentiles: list[int]
    :ivar top_tag_count: Maximum number of tags to include in top tag output.
    :vartype top_tag_count: int
    :ivar tag_filters: Optional tag filters to limit tag coverage metrics.
    :vartype tag_filters: list[str] or None
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    sample_size: Optional[int] = Field(default=None, ge=1)
    min_text_characters: Optional[int] = Field(default=None, ge=1)
    percentiles: List[int] = Field(default_factory=lambda: [50, 90, 99])
    top_tag_count: int = Field(default=10, ge=1)
    tag_filters: Optional[List[str]] = None

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "ProfilingConfiguration":
        if self.schema_version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported analysis schema version: {self.schema_version}")
        return self

    @field_validator("percentiles", mode="after")
    @classmethod
    def _validate_percentiles(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("percentiles must include at least one value")
        if any(percentile < 1 or percentile > 100 for percentile in value):
            raise ValueError("percentiles must be between 1 and 100")
        if value != sorted(value):
            raise ValueError("percentiles must be sorted in ascending order")
        return value

    @field_validator("tag_filters", mode="before")
    @classmethod
    def _validate_tag_filters(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("tag_filters must be a list of strings")
        cleaned = [str(tag).strip() for tag in value]
        if not cleaned or any(not tag for tag in cleaned):
            raise ValueError("tag_filters must be a list of non-empty strings")
        return cleaned


class ProfilingPercentileValue(AnalysisSchemaModel):
    """
    Percentile entry for a distribution.

    :ivar percentile: Percentile value between 1 and 100.
    :vartype percentile: int
    :ivar value: Percentile value.
    :vartype value: float
    """

    percentile: int = Field(ge=1, le=100)
    value: float


class ProfilingDistributionReport(AnalysisSchemaModel):
    """
    Distribution summary for numeric values.

    :ivar count: Count of values included.
    :vartype count: int
    :ivar min_value: Minimum value observed.
    :vartype min_value: float
    :ivar max_value: Maximum value observed.
    :vartype max_value: float
    :ivar mean_value: Mean value observed.
    :vartype mean_value: float
    :ivar percentiles: Percentile values.
    :vartype percentiles: list[ProfilingPercentileValue]
    """

    count: int = Field(ge=0)
    min_value: float
    max_value: float
    mean_value: float
    percentiles: List[ProfilingPercentileValue] = Field(default_factory=list)


class ProfilingTagCount(AnalysisSchemaModel):
    """
    Tag count entry for profiling output.

    :ivar tag: Tag name.
    :vartype tag: str
    :ivar count: Number of items with this tag.
    :vartype count: int
    """

    tag: str
    count: int = Field(ge=0)


class ProfilingTagReport(AnalysisSchemaModel):
    """
    Tag coverage summary for raw items.

    :ivar tagged_items: Count of items with tags.
    :vartype tagged_items: int
    :ivar untagged_items: Count of items without tags.
    :vartype untagged_items: int
    :ivar total_unique_tags: Count of unique tags.
    :vartype total_unique_tags: int
    :ivar top_tags: Most frequent tags.
    :vartype top_tags: list[ProfilingTagCount]
    :ivar tag_filters: Optional tag filters applied.
    :vartype tag_filters: list[str] or None
    """

    tagged_items: int = Field(ge=0)
    untagged_items: int = Field(ge=0)
    total_unique_tags: int = Field(ge=0)
    top_tags: List[ProfilingTagCount] = Field(default_factory=list)
    tag_filters: Optional[List[str]] = None


class ProfilingRawItemsReport(AnalysisSchemaModel):
    """
    Summary of raw corpus items.

    :ivar total_items: Total number of catalog items.
    :vartype total_items: int
    :ivar media_type_counts: Count of items per media type.
    :vartype media_type_counts: dict[str, int]
    :ivar bytes_distribution: Distribution of raw item sizes in bytes.
    :vartype bytes_distribution: ProfilingDistributionReport
    :ivar tags: Tag coverage summary.
    :vartype tags: ProfilingTagReport
    """

    total_items: int = Field(ge=0)
    media_type_counts: Dict[str, int] = Field(default_factory=dict)
    bytes_distribution: ProfilingDistributionReport
    tags: ProfilingTagReport


class ProfilingExtractedTextReport(AnalysisSchemaModel):
    """
    Summary of extracted text coverage.

    :ivar source_items: Count of source items in the extraction snapshot.
    :vartype source_items: int
    :ivar extracted_nonempty_items: Count of extracted items with non-empty text.
    :vartype extracted_nonempty_items: int
    :ivar extracted_empty_items: Count of extracted items with empty text.
    :vartype extracted_empty_items: int
    :ivar extracted_missing_items: Count of items with no extracted text artifact.
    :vartype extracted_missing_items: int
    :ivar characters_distribution: Distribution of extracted text lengths.
    :vartype characters_distribution: ProfilingDistributionReport
    """

    source_items: int = Field(ge=0)
    extracted_nonempty_items: int = Field(ge=0)
    extracted_empty_items: int = Field(ge=0)
    extracted_missing_items: int = Field(ge=0)
    characters_distribution: ProfilingDistributionReport


class ProfilingReport(AnalysisSchemaModel):
    """
    Report for profiling analysis.

    :ivar raw_items: Raw corpus item summary.
    :vartype raw_items: ProfilingRawItemsReport
    :ivar extracted_text: Extracted text coverage summary.
    :vartype extracted_text: ProfilingExtractedTextReport
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    raw_items: ProfilingRawItemsReport
    extracted_text: ProfilingExtractedTextReport
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ProfilingOutput(AnalysisSchemaModel):
    """
    Output bundle for profiling analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar analysis_id: Analysis backend identifier.
    :vartype analysis_id: str
    :ivar generated_at: International Organization for Standardization 8601 timestamp for output creation.
    :vartype generated_at: str
    :ivar snapshot: Analysis snapshot manifest.
    :vartype snapshot: AnalysisRunManifest
    :ivar report: Profiling report data.
    :vartype report: ProfilingReport
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    analysis_id: str
    generated_at: str
    snapshot: AnalysisRunManifest
    report: ProfilingReport


class TopicModelingTextSourceConfig(AnalysisSchemaModel):
    """
    Configuration for text collection within topic modeling.

    :ivar sample_size: Optional sample size for text collection.
    :vartype sample_size: int or None
    :ivar min_text_characters: Optional minimum character count for text inclusion.
    :vartype min_text_characters: int or None
    """

    sample_size: Optional[int] = Field(default=None, ge=1)
    min_text_characters: Optional[int] = Field(default=None, ge=1)


class TopicModelingLlmExtractionMethod(str, Enum):
    """
    LLM extraction method identifiers.
    """

    SINGLE = "single"
    ITEMIZE = "itemize"


class TopicModelingLlmExtractionConfig(AnalysisSchemaModel):
    """
    Configuration for LLM-based extraction within topic modeling.

    :ivar enabled: Whether LLM extraction is enabled.
    :vartype enabled: bool
    :ivar method: Extraction method, single or itemize.
    :vartype method: TopicModelingLlmExtractionMethod
    :ivar client: LLM client configuration.
    :vartype client: LlmClientConfig or None
    :ivar prompt_template: Prompt template containing the {text} placeholder.
    :vartype prompt_template: str or None
    :ivar system_prompt: Optional system prompt.
    :vartype system_prompt: str or None
    :ivar max_workers: Maximum concurrent LLM extraction workers.
    :vartype max_workers: int
    """

    enabled: bool = Field(default=False)
    method: TopicModelingLlmExtractionMethod = Field(
        default=TopicModelingLlmExtractionMethod.SINGLE
    )
    client: Optional[LlmClientConfig] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    max_workers: int = Field(default=1, ge=1)

    @field_validator("method", mode="before")
    @classmethod
    def _parse_method(cls, value: object) -> TopicModelingLlmExtractionMethod:
        if isinstance(value, TopicModelingLlmExtractionMethod):
            return value
        if isinstance(value, str):
            return TopicModelingLlmExtractionMethod(value)
        raise ValueError(
            "llm_extraction.method must be a string or TopicModelingLlmExtractionMethod"
        )

    @model_validator(mode="after")
    def _validate_requirements(self) -> "TopicModelingLlmExtractionConfig":
        if not self.enabled:
            return self
        if self.client is None:
            raise ValueError("llm_extraction.client is required when enabled")
        if self.prompt_template is None:
            raise ValueError("llm_extraction.prompt_template is required when enabled")
        if "{text}" not in self.prompt_template:
            raise ValueError("llm_extraction.prompt_template must include {text}")
        return self


class TopicModelingLexicalProcessingConfig(AnalysisSchemaModel):
    """
    Configuration for lexical processing within topic modeling.

    :ivar enabled: Whether lexical processing is enabled.
    :vartype enabled: bool
    :ivar lowercase: Whether to lowercase text.
    :vartype lowercase: bool
    :ivar strip_punctuation: Whether to remove punctuation.
    :vartype strip_punctuation: bool
    :ivar collapse_whitespace: Whether to normalize whitespace.
    :vartype collapse_whitespace: bool
    """

    enabled: bool = Field(default=False)
    lowercase: bool = Field(default=True)
    strip_punctuation: bool = Field(default=False)
    collapse_whitespace: bool = Field(default=True)


class TopicModelingEntityRemovalConfig(AnalysisSchemaModel):
    """
    Configuration for entity removal preprocessing.

    :ivar enabled: Whether entity removal is enabled.
    :vartype enabled: bool
    :ivar provider: Entity removal provider identifier (currently ``spacy``).
    :vartype provider: str
    :ivar model: Provider model name (for example: ``en_core_web_sm``).
    :vartype model: str
    :ivar entity_types: Optional list of entity labels to remove. When empty, defaults are used.
    :vartype entity_types: list[str]
    :ivar replace_with: Replacement text inserted for removed entities (empty string removes).
    :vartype replace_with: str
    :ivar collapse_whitespace: Whether to collapse whitespace after removals.
    :vartype collapse_whitespace: bool
    :ivar regex_patterns: Optional list of regular expressions to remove after NER.
    :vartype regex_patterns: list[str]
    :ivar regex_replace_with: Replacement text for regex removals.
    :vartype regex_replace_with: str
    """

    enabled: bool = Field(default=False)
    provider: str = Field(default="spacy", min_length=1)
    model: str = Field(default="en_core_web_sm", min_length=1)
    entity_types: List[str] = Field(default_factory=list)
    replace_with: str = Field(default="")
    collapse_whitespace: bool = Field(default=True)
    regex_patterns: List[str] = Field(default_factory=list)
    regex_replace_with: str = Field(default="")

    @model_validator(mode="after")
    def _validate_requirements(self) -> "TopicModelingEntityRemovalConfig":
        if not self.enabled:
            return self
        if self.provider.strip().lower() != "spacy":
            raise ValueError("entity_removal.provider must be 'spacy'")
        return self


class TopicModelingVectorizerConfig(AnalysisSchemaModel):
    """
    Vectorizer configuration for BERTopic tokenization.

    :ivar ngram_range: Inclusive n-gram range as a two-item list.
    :vartype ngram_range: list[int]
    :ivar stop_words: Stop word configuration for tokenization.
    :vartype stop_words: str or list[str] or None
    """

    ngram_range: List[int] = Field(default_factory=lambda: [1, 1], min_length=2, max_length=2)
    stop_words: Optional[object] = None

    @model_validator(mode="after")
    def _validate_ngram_range(self) -> "TopicModelingVectorizerConfig":
        start, end = self.ngram_range
        if start < 1 or end < start:
            raise ValueError(
                "vectorizer.ngram_range must include two integers with start >= 1 and end >= start"
            )
        return self

    @field_validator("stop_words", mode="before")
    @classmethod
    def _validate_stop_words(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            if value != "english":
                raise ValueError("vectorizer.stop_words must be 'english' or a list of strings")
            return value
        if isinstance(value, list):
            if not value or not all(isinstance(entry, str) and entry for entry in value):
                raise ValueError("vectorizer.stop_words must be 'english' or a list of strings")
            return value
        raise ValueError("vectorizer.stop_words must be 'english' or a list of strings")


class TopicModelingBerTopicConfig(AnalysisSchemaModel):
    """
    Configuration for BERTopic analysis.

    :ivar parameters: Parameters forwarded to the BERTopic constructor.
    :vartype parameters: dict[str, Any]
    :ivar vectorizer: Vectorizer configuration for tokenization.
    :vartype vectorizer: TopicModelingVectorizerConfig or None
    """

    parameters: Dict[str, Any] = Field(default_factory=dict)
    vectorizer: Optional[TopicModelingVectorizerConfig] = None


class TopicModelingLlmFineTuningConfig(AnalysisSchemaModel):
    """
    Configuration for LLM-based topic labeling.

    :ivar enabled: Whether LLM topic labeling is enabled.
    :vartype enabled: bool
    :ivar client: LLM client configuration.
    :vartype client: LlmClientConfig or None
    :ivar prompt_template: Prompt template containing {keywords} and {documents} placeholders.
    :vartype prompt_template: str or None
    :ivar system_prompt: Optional system prompt.
    :vartype system_prompt: str or None
    :ivar max_keywords: Maximum number of keywords to include in prompts.
    :vartype max_keywords: int
    :ivar max_documents: Maximum number of documents to include in prompts.
    :vartype max_documents: int
    """

    enabled: bool = Field(default=False)
    client: Optional[LlmClientConfig] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    max_keywords: int = Field(default=8, ge=1)
    max_documents: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _validate_requirements(self) -> "TopicModelingLlmFineTuningConfig":
        if not self.enabled:
            return self
        if self.client is None:
            raise ValueError("llm_fine_tuning.client is required when enabled")
        if self.prompt_template is None:
            raise ValueError("llm_fine_tuning.prompt_template is required when enabled")
        if "{keywords}" not in self.prompt_template or "{documents}" not in self.prompt_template:
            raise ValueError(
                "llm_fine_tuning.prompt_template must include {keywords} and {documents}"
            )
        return self


class TopicModelingConfiguration(AnalysisSchemaModel):
    """
    Configuration for topic modeling analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar text_source: Text collection configuration.
    :vartype text_source: TopicModelingTextSourceConfig
    :ivar llm_extraction: LLM extraction configuration.
    :vartype llm_extraction: TopicModelingLlmExtractionConfig
    :ivar entity_removal: Entity removal preprocessing configuration.
    :vartype entity_removal: TopicModelingEntityRemovalConfig
    :ivar lexical_processing: Lexical processing configuration.
    :vartype lexical_processing: TopicModelingLexicalProcessingConfig
    :ivar bertopic_analysis: BERTopic configuration.
    :vartype bertopic_analysis: TopicModelingBerTopicConfig
    :ivar llm_fine_tuning: LLM fine-tuning configuration.
    :vartype llm_fine_tuning: TopicModelingLlmFineTuningConfig
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    text_source: TopicModelingTextSourceConfig = Field(
        default_factory=TopicModelingTextSourceConfig
    )
    llm_extraction: TopicModelingLlmExtractionConfig = Field(
        default_factory=TopicModelingLlmExtractionConfig
    )
    entity_removal: TopicModelingEntityRemovalConfig = Field(
        default_factory=TopicModelingEntityRemovalConfig
    )
    lexical_processing: TopicModelingLexicalProcessingConfig = Field(
        default_factory=TopicModelingLexicalProcessingConfig
    )
    bertopic_analysis: TopicModelingBerTopicConfig = Field(
        default_factory=TopicModelingBerTopicConfig
    )
    llm_fine_tuning: TopicModelingLlmFineTuningConfig = Field(
        default_factory=TopicModelingLlmFineTuningConfig
    )

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "TopicModelingConfiguration":
        if self.schema_version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported analysis schema version: {self.schema_version}")
        return self


class TopicModelingStageStatus(str, Enum):
    """
    Stage status values for topic modeling.
    """

    COMPLETE = "complete"
    SKIPPED = "skipped"
    FAILED = "failed"


class TopicModelingTextCollectionReport(AnalysisSchemaModel):
    """
    Report for the text collection stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar source_items: Count of source items inspected.
    :vartype source_items: int
    :ivar documents: Count of documents produced.
    :vartype documents: int
    :ivar sample_size: Optional sample size.
    :vartype sample_size: int or None
    :ivar min_text_characters: Optional minimum character threshold.
    :vartype min_text_characters: int or None
    :ivar empty_texts: Count of empty text inputs.
    :vartype empty_texts: int
    :ivar skipped_items: Count of skipped items.
    :vartype skipped_items: int
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: TopicModelingStageStatus
    source_items: int = Field(ge=0)
    documents: int = Field(ge=0)
    sample_size: Optional[int] = None
    min_text_characters: Optional[int] = None
    empty_texts: int = Field(ge=0)
    skipped_items: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingLlmExtractionReport(AnalysisSchemaModel):
    """
    Report for the LLM extraction stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar method: Extraction method used.
    :vartype method: TopicModelingLlmExtractionMethod
    :ivar input_documents: Count of input documents.
    :vartype input_documents: int
    :ivar output_documents: Count of output documents.
    :vartype output_documents: int
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: TopicModelingStageStatus
    method: TopicModelingLlmExtractionMethod
    input_documents: int = Field(ge=0)
    output_documents: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingEntityRemovalReport(AnalysisSchemaModel):
    """
    Report for the entity removal stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar provider: Entity removal provider identifier.
    :vartype provider: str
    :ivar model: Provider model name.
    :vartype model: str
    :ivar entity_types: Entity labels targeted for removal.
    :vartype entity_types: list[str]
    :ivar input_documents: Count of input documents.
    :vartype input_documents: int
    :ivar output_documents: Count of output documents.
    :vartype output_documents: int
    :ivar regex_patterns: Regular expressions applied after NER.
    :vartype regex_patterns: list[str]
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: TopicModelingStageStatus
    provider: str
    model: str
    entity_types: List[str] = Field(default_factory=list)
    input_documents: int = Field(ge=0)
    output_documents: int = Field(ge=0)
    regex_patterns: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingLexicalProcessingReport(AnalysisSchemaModel):
    """
    Report for the lexical processing stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar input_documents: Count of input documents.
    :vartype input_documents: int
    :ivar output_documents: Count of output documents.
    :vartype output_documents: int
    :ivar lowercase: Whether lowercase normalization was applied.
    :vartype lowercase: bool
    :ivar strip_punctuation: Whether punctuation was removed.
    :vartype strip_punctuation: bool
    :ivar collapse_whitespace: Whether whitespace was normalized.
    :vartype collapse_whitespace: bool
    """

    status: TopicModelingStageStatus
    input_documents: int = Field(ge=0)
    output_documents: int = Field(ge=0)
    lowercase: bool
    strip_punctuation: bool
    collapse_whitespace: bool


class TopicModelingBerTopicReport(AnalysisSchemaModel):
    """
    Report for the BERTopic analysis stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar topic_count: Count of topics discovered.
    :vartype topic_count: int
    :ivar document_count: Count of documents analyzed.
    :vartype document_count: int
    :ivar parameters: BERTopic configuration parameters.
    :vartype parameters: dict[str, Any]
    :ivar vectorizer: Vectorizer configuration applied to BERTopic.
    :vartype vectorizer: TopicModelingVectorizerConfig or None
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: TopicModelingStageStatus
    topic_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    vectorizer: Optional[TopicModelingVectorizerConfig] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingLlmFineTuningReport(AnalysisSchemaModel):
    """
    Report for the LLM fine-tuning stage.

    :ivar status: Stage status.
    :vartype status: TopicModelingStageStatus
    :ivar topics_labeled: Count of topics labeled.
    :vartype topics_labeled: int
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: TopicModelingStageStatus
    topics_labeled: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingLabelSource(str, Enum):
    """
    Source identifiers for topic labels.
    """

    BERTOPIC = "bertopic"
    LLM = "llm"


class TopicModelingKeyword(AnalysisSchemaModel):
    """
    Keyword entry for a topic.

    :ivar keyword: Keyword or phrase.
    :vartype keyword: str
    :ivar score: Keyword relevance score.
    :vartype score: float
    """

    keyword: str
    score: float


class TopicModelingTopic(AnalysisSchemaModel):
    """
    Topic output record.

    :ivar topic_id: Topic identifier.
    :vartype topic_id: int
    :ivar label: Human-readable topic label.
    :vartype label: str
    :ivar label_source: Source for the label.
    :vartype label_source: TopicModelingLabelSource
    :ivar keywords: Topic keywords with scores.
    :vartype keywords: list[TopicModelingKeyword]
    :ivar document_count: Number of documents in the topic.
    :vartype document_count: int
    :ivar document_examples: Example document texts.
    :vartype document_examples: list[str]
    :ivar document_ids: Document identifiers for the topic.
    :vartype document_ids: list[str]
    """

    topic_id: int
    label: str
    label_source: TopicModelingLabelSource
    keywords: List[TopicModelingKeyword] = Field(default_factory=list)
    document_count: int = Field(ge=0)
    document_examples: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)


class TopicModelingReport(AnalysisSchemaModel):
    """
    Report for topic modeling analysis.

    :ivar text_collection: Text collection report.
    :vartype text_collection: TopicModelingTextCollectionReport
    :ivar llm_extraction: LLM extraction report.
    :vartype llm_extraction: TopicModelingLlmExtractionReport
    :ivar entity_removal: Entity removal report.
    :vartype entity_removal: TopicModelingEntityRemovalReport
    :ivar lexical_processing: Lexical processing report.
    :vartype lexical_processing: TopicModelingLexicalProcessingReport
    :ivar bertopic_analysis: BERTopic analysis report.
    :vartype bertopic_analysis: TopicModelingBerTopicReport
    :ivar llm_fine_tuning: LLM fine-tuning report.
    :vartype llm_fine_tuning: TopicModelingLlmFineTuningReport
    :ivar topics: Topic output list.
    :vartype topics: list[TopicModelingTopic]
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    text_collection: TopicModelingTextCollectionReport
    llm_extraction: TopicModelingLlmExtractionReport
    entity_removal: TopicModelingEntityRemovalReport
    lexical_processing: TopicModelingLexicalProcessingReport
    bertopic_analysis: TopicModelingBerTopicReport
    llm_fine_tuning: TopicModelingLlmFineTuningReport
    topics: List[TopicModelingTopic] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class TopicModelingOutput(AnalysisSchemaModel):
    """
    Output bundle for topic modeling analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar analysis_id: Analysis backend identifier.
    :vartype analysis_id: str
    :ivar generated_at: International Organization for Standardization 8601 timestamp for output creation.
    :vartype generated_at: str
    :ivar snapshot: Analysis snapshot manifest.
    :vartype snapshot: AnalysisRunManifest
    :ivar report: Topic modeling report data.
    :vartype report: TopicModelingReport
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    analysis_id: str
    generated_at: str
    snapshot: AnalysisRunManifest
    report: TopicModelingReport


class MarkovAnalysisStageStatus(str, Enum):
    """
    Status values for Markov analysis stages.
    """

    SKIPPED = "skipped"
    COMPLETE = "complete"
    FAILED = "failed"


class MarkovAnalysisSegmentationMethod(str, Enum):
    """
    Segmentation method identifiers for Markov analysis.
    """

    SENTENCE = "sentence"
    FIXED_WINDOW = "fixed_window"
    LLM = "llm"
    SPAN_MARKUP = "span_markup"


class MarkovAnalysisLlmSegmentationConfig(AnalysisSchemaModel):
    """
    Provider-backed segmentation configuration.

    :ivar client: LLM client configuration.
    :vartype client: biblicus.ai.models.LlmClientConfig
    :ivar prompt_template: Prompt template containing ``{text}``.
    :vartype prompt_template: str
    :ivar system_prompt: Optional system prompt.
    :vartype system_prompt: str or None
    """

    client: LlmClientConfig
    prompt_template: str = Field(min_length=1)
    system_prompt: Optional[str] = None


class MarkovAnalysisSpanMarkupSegmentationConfig(AnalysisSchemaModel):
    """
    Provider-backed text extract configuration.

    :ivar client: LLM client configuration.
    :vartype client: biblicus.ai.models.LlmClientConfig
    :ivar prompt_template: Prompt template describing what to return (must not include ``{text}``).
    :vartype prompt_template: str
    :ivar system_prompt: Optional system prompt containing ``{text}``.
    :vartype system_prompt: str or None
    :ivar max_rounds: Maximum number of edit rounds.
    :vartype max_rounds: int
    :ivar max_edits_per_round: Maximum edits per round.
    :vartype max_edits_per_round: int
    :ivar normalize_nested_spans: Whether to normalize nested spans from text extract output.
    :vartype normalize_nested_spans: bool
    :ivar chunk_characters: Optional chunk size for span markup processing.
    :vartype chunk_characters: int or None
    :ivar chunk_overlap_characters: Optional overlap between chunks.
    :vartype chunk_overlap_characters: int or None
    :ivar label_attribute: Optional attribute name used to extract segment labels.
    :vartype label_attribute: str or None
    :ivar prepend_label: Whether to prepend the label and a newline to segment text.
    :vartype prepend_label: bool
    :ivar start_label_value: Optional marker prepended to the first segment.
    :vartype start_label_value: str or None
    :ivar end_label_value: Optional marker prepended to the last segment when verified.
    :vartype end_label_value: str or None
    :ivar end_label_verifier: Optional LLM verifier for end-label assignment.
    :vartype end_label_verifier: MarkovAnalysisSpanMarkupEndLabelVerifierConfig or None
    :ivar end_reject_label_value: Optional marker prepended when the verifier rejects an end label.
    :vartype end_reject_label_value: str or None
    :ivar end_reject_reason_prefix: Prefix used for the verifier explanation line.
    :vartype end_reject_reason_prefix: str
    """

    client: LlmClientConfig
    prompt_template: str = Field(min_length=1)
    system_prompt: Optional[str] = None
    max_rounds: int = Field(default=6, ge=1)
    max_edits_per_round: int = Field(default=500, ge=1)
    normalize_nested_spans: bool = Field(default=False)
    chunk_characters: Optional[int] = Field(default=None, ge=1)
    chunk_overlap_characters: Optional[int] = Field(default=None, ge=0)
    label_attribute: Optional[str] = Field(default=None, min_length=1)
    prepend_label: bool = False
    start_label_value: Optional[str] = Field(default=None, min_length=1)
    end_label_value: Optional[str] = Field(default=None, min_length=1)
    end_label_verifier: Optional["MarkovAnalysisSpanMarkupEndLabelVerifierConfig"] = None
    end_reject_label_value: Optional[str] = Field(default=None, min_length=1)
    end_reject_reason_prefix: str = Field(default="disconnection_reason", min_length=1)

    @model_validator(mode="after")
    def _validate_prompt_template(self) -> "MarkovAnalysisSpanMarkupSegmentationConfig":
        if self.system_prompt is not None and "{text}" not in self.system_prompt:
            raise ValueError("segmentation.span_markup.system_prompt must include {text}")
        if "{text}" in self.prompt_template:
            raise ValueError("segmentation.span_markup.prompt_template must not include {text}")
        if self.chunk_overlap_characters is not None and self.chunk_characters is None:
            raise ValueError(
                "segmentation.span_markup.chunk_overlap_characters requires chunk_characters"
            )
        if (
            self.chunk_characters is not None
            and self.chunk_overlap_characters is not None
            and self.chunk_overlap_characters >= self.chunk_characters
        ):
            raise ValueError(
                "segmentation.span_markup.chunk_overlap_characters must be smaller than chunk_characters"
            )
        if self.prepend_label and not self.label_attribute:
            raise ValueError(
                "segmentation.span_markup.label_attribute is required when "
                "segmentation.span_markup.prepend_label is true"
            )
        if self.end_label_value is not None and self.end_label_verifier is None:
            raise ValueError(
                "segmentation.span_markup.end_label_verifier is required when "
                "segmentation.span_markup.end_label_value is set"
            )
        if self.end_reject_label_value is not None and self.end_label_verifier is None:
            raise ValueError(
                "segmentation.span_markup.end_label_verifier is required when "
                "segmentation.span_markup.end_reject_label_value is set"
            )
        return self


class MarkovAnalysisSpanMarkupEndLabelVerifierConfig(AnalysisSchemaModel):
    """
    Verifier configuration for end-label assignment.

    :ivar client: LLM client configuration.
    :vartype client: biblicus.ai.models.LlmClientConfig
    :ivar system_prompt: System prompt containing ``{text}``.
    :vartype system_prompt: str
    :ivar prompt_template: Prompt template for the verifier (must not include ``{text}``).
    :vartype prompt_template: str
    """

    client: LlmClientConfig
    system_prompt: str = Field(min_length=1)
    prompt_template: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_prompt_template(
        self,
    ) -> "MarkovAnalysisSpanMarkupEndLabelVerifierConfig":
        if "{text}" not in self.system_prompt:
            raise ValueError(
                "segmentation.span_markup.end_label_verifier.system_prompt must include {text}"
            )
        if "{text}" in self.prompt_template:
            raise ValueError(
                "segmentation.span_markup.end_label_verifier.prompt_template must not include {text}"
            )
        return self


class MarkovAnalysisTextSourceConfig(AnalysisSchemaModel):
    """
    Text source configuration for Markov analysis.

    :ivar sample_size: Optional cap on number of documents included.
    :vartype sample_size: int or None
    :ivar min_text_characters: Optional minimum extracted text length.
    :vartype min_text_characters: int or None
    """

    sample_size: Optional[int] = Field(default=None, ge=1)
    min_text_characters: Optional[int] = Field(default=None, ge=1)


class MarkovAnalysisFixedWindowSegmentationConfig(AnalysisSchemaModel):
    """
    Fixed window segmentation configuration.

    :ivar max_characters: Maximum segment size in characters.
    :vartype max_characters: int
    :ivar overlap_characters: Overlap between consecutive segments.
    :vartype overlap_characters: int
    """

    max_characters: int = Field(default=800, ge=1)
    overlap_characters: int = Field(default=0, ge=0)


class MarkovAnalysisSegmentationConfig(AnalysisSchemaModel):
    """
    Segmentation configuration for Markov analysis.

    :ivar method: Segmentation method identifier.
    :vartype method: MarkovAnalysisSegmentationMethod
    :ivar fixed_window: Fixed window settings for ``fixed_window`` method.
    :vartype fixed_window: MarkovAnalysisFixedWindowSegmentationConfig
    :ivar span_markup: Text extract settings for ``span_markup`` method.
    :vartype span_markup: MarkovAnalysisSpanMarkupSegmentationConfig or None
    """

    method: MarkovAnalysisSegmentationMethod = Field(
        default=MarkovAnalysisSegmentationMethod.SENTENCE
    )
    max_workers: int = Field(default=1, ge=1)
    fixed_window: MarkovAnalysisFixedWindowSegmentationConfig = Field(
        default_factory=MarkovAnalysisFixedWindowSegmentationConfig
    )
    llm: Optional[MarkovAnalysisLlmSegmentationConfig] = None
    span_markup: Optional[MarkovAnalysisSpanMarkupSegmentationConfig] = None

    @field_validator("method", mode="before")
    @classmethod
    def _parse_method(cls, value: object) -> MarkovAnalysisSegmentationMethod:
        if isinstance(value, MarkovAnalysisSegmentationMethod):
            return value
        if isinstance(value, str):
            return MarkovAnalysisSegmentationMethod(value)
        raise ValueError("segmentation.method must be a string or MarkovAnalysisSegmentationMethod")

    @model_validator(mode="after")
    def _validate_requirements(self) -> "MarkovAnalysisSegmentationConfig":
        if self.method == MarkovAnalysisSegmentationMethod.LLM and self.llm is None:
            raise ValueError("segmentation.llm is required when segmentation.method is 'llm'")
        if self.method == MarkovAnalysisSegmentationMethod.SPAN_MARKUP and self.span_markup is None:
            raise ValueError(
                "segmentation.span_markup is required when segmentation.method is 'span_markup'"
            )
        return self


class MarkovAnalysisLlmObservationsCacheConfig(AnalysisSchemaModel):
    """
    Cache configuration for LLM observations.

    :ivar enabled: Whether to reuse cached LLM observations.
    :vartype enabled: bool
    :ivar cache_name: Optional cache name used in cache identity.
    :vartype cache_name: str or None
    """

    enabled: bool = Field(default=True)
    cache_name: Optional[str] = Field(default=None, min_length=1)


class MarkovAnalysisLlmObservationsConfig(AnalysisSchemaModel):
    """
    Provider-backed observation extraction configuration.

    :ivar enabled: Whether to enable provider-backed observation extraction.
    :vartype enabled: bool
    :ivar cache: Cache configuration for LLM observation results.
    :vartype cache: MarkovAnalysisLlmObservationsCacheConfig
    :ivar client: LLM client configuration.
    :vartype client: biblicus.ai.models.LlmClientConfig
    :ivar prompt_template: Prompt template containing ``{segment}``.
    :vartype prompt_template: str
    :ivar system_prompt: Optional system prompt.
    :vartype system_prompt: str or None
    """

    enabled: bool = Field(default=False)
    cache: MarkovAnalysisLlmObservationsCacheConfig = Field(
        default_factory=MarkovAnalysisLlmObservationsCacheConfig
    )
    client: Optional[LlmClientConfig] = None
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    max_workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_requirements(self) -> "MarkovAnalysisLlmObservationsConfig":
        if not self.enabled:
            return self
        if self.client is None:
            raise ValueError(
                "llm_observations.client is required when llm_observations.enabled is true"
            )
        if not self.prompt_template:
            raise ValueError(
                "llm_observations.prompt_template is required when llm_observations.enabled is true"
            )
        return self


class MarkovAnalysisEmbeddingsConfig(AnalysisSchemaModel):
    """
    Provider-backed embeddings configuration.

    :ivar enabled: Whether to generate embeddings.
    :vartype enabled: bool
    :ivar client: Embeddings client configuration.
    :vartype client: biblicus.ai.models.EmbeddingsClientConfig
    :ivar text_source: Which text field to embed (``segment_text`` or ``llm_summary``).
    :vartype text_source: str
    """

    enabled: bool = Field(default=False)
    client: Optional[EmbeddingsClientConfig] = None
    text_source: str = Field(default="segment_text", min_length=1)

    @model_validator(mode="after")
    def _validate_requirements(self) -> "MarkovAnalysisEmbeddingsConfig":
        if not self.enabled:
            return self
        if self.client is None:
            raise ValueError("embeddings.client is required when embeddings.enabled is true")
        if self.text_source not in {"segment_text", "llm_summary"}:
            raise ValueError("embeddings.text_source must be 'segment_text' or 'llm_summary'")
        return self


class MarkovAnalysisTopicModelingConfig(AnalysisSchemaModel):
    """
    Topic modeling configuration for Markov analysis observations.

    :ivar enabled: Whether to run topic modeling on segments.
    :vartype enabled: bool
    :ivar configuration: Topic modeling configuration applied to segments.
    :vartype configuration: TopicModelingConfiguration or None
    """

    enabled: bool = Field(default=False)
    configuration: Optional["TopicModelingConfiguration"] = None

    @model_validator(mode="after")
    def _validate_requirements(self) -> "MarkovAnalysisTopicModelingConfig":
        if not self.enabled:
            return self
        if self.configuration is None:
            raise ValueError(
                "topic_modeling.configuration is required when topic_modeling.enabled is true"
            )
        if self.configuration.llm_extraction.enabled and (
            self.configuration.llm_extraction.method != TopicModelingLlmExtractionMethod.SINGLE
        ):
            raise ValueError(
                "topic_modeling.configuration.llm_extraction.method must be 'single' for Markov topic modeling"
            )
        return self


class MarkovAnalysisObservationsEncoder(str, Enum):
    """
    Observation encoder identifiers.
    """

    TFIDF = "tfidf"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class MarkovAnalysisTfidfObservationConfig(AnalysisSchemaModel):
    """
    TF-IDF encoder configuration for local observations.

    :ivar max_features: Maximum vocabulary size.
    :vartype max_features: int
    :ivar ngram_range: Inclusive n-gram range.
    :vartype ngram_range: list[int]
    """

    max_features: int = Field(default=2000, ge=1)
    ngram_range: List[int] = Field(default_factory=lambda: [1, 2])

    @field_validator("ngram_range", mode="before")
    @classmethod
    def _validate_ngram_range(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("tfidf.ngram_range must be a list of two integers")
        if any(not isinstance(item, int) for item in value):
            raise ValueError("tfidf.ngram_range must be a list of two integers")
        if value[0] < 1 or value[1] < value[0]:
            raise ValueError("tfidf.ngram_range must be a valid inclusive range")
        return value


class MarkovAnalysisObservationsConfig(AnalysisSchemaModel):
    """
    Observations configuration for Markov analysis.

    :ivar encoder: Observation encoder identifier.
    :vartype encoder: MarkovAnalysisObservationsEncoder
    :ivar tfidf: TF-IDF encoder settings.
    :vartype tfidf: MarkovAnalysisTfidfObservationConfig
    :ivar text_source: Which text field to encode for ``tfidf`` (``segment_text`` or ``llm_summary``).
    :vartype text_source: str
    :ivar categorical_source: Which field provides categorical labels for hybrid/categorical use.
    :vartype categorical_source: str
    :ivar numeric_source: Which field provides a numeric scalar feature for hybrid use.
    :vartype numeric_source: str
    """

    encoder: MarkovAnalysisObservationsEncoder = Field(
        default=MarkovAnalysisObservationsEncoder.TFIDF
    )
    tfidf: MarkovAnalysisTfidfObservationConfig = Field(
        default_factory=MarkovAnalysisTfidfObservationConfig
    )
    text_source: str = Field(default="segment_text", min_length=1)
    categorical_source: str = Field(default="llm_label", min_length=1)
    numeric_source: str = Field(default="llm_label_confidence", min_length=1)

    @field_validator("encoder", mode="before")
    @classmethod
    def _parse_encoder(cls, value: object) -> MarkovAnalysisObservationsEncoder:
        if isinstance(value, MarkovAnalysisObservationsEncoder):
            return value
        if isinstance(value, str):
            return MarkovAnalysisObservationsEncoder(value)
        raise ValueError(
            "observations.encoder must be a string or MarkovAnalysisObservationsEncoder"
        )

    @model_validator(mode="after")
    def _validate_sources(self) -> "MarkovAnalysisObservationsConfig":
        if self.text_source not in {"segment_text", "llm_summary"}:
            raise ValueError("observations.text_source must be 'segment_text' or 'llm_summary'")
        return self


class MarkovAnalysisModelFamily(str, Enum):
    """
    Markov model family identifiers.
    """

    GAUSSIAN = "gaussian"
    CATEGORICAL = "categorical"


class MarkovAnalysisModelConfig(AnalysisSchemaModel):
    """
    Model configuration for Markov analysis.

    :ivar family: Model family identifier.
    :vartype family: MarkovAnalysisModelFamily
    :ivar n_states: Number of hidden states to learn.
    :vartype n_states: int
    """

    family: MarkovAnalysisModelFamily = Field(default=MarkovAnalysisModelFamily.GAUSSIAN)
    n_states: int = Field(default=8, ge=1)

    @field_validator("family", mode="before")
    @classmethod
    def _parse_family(cls, value: object) -> MarkovAnalysisModelFamily:
        if isinstance(value, MarkovAnalysisModelFamily):
            return value
        if isinstance(value, str):
            return MarkovAnalysisModelFamily(value)
        raise ValueError("model.family must be a string or MarkovAnalysisModelFamily")


class MarkovAnalysisArtifactsGraphVizConfig(AnalysisSchemaModel):
    """
    GraphViz export configuration.

    :ivar enabled: Whether to write GraphViz transitions output.
    :vartype enabled: bool
    :ivar rankdir: GraphViz rank direction (e.g., LR or TB).
    :vartype rankdir: str
    :ivar min_edge_weight: Minimum edge weight to include in GraphViz output.
    :vartype min_edge_weight: float
    :ivar start_state_id: Optional state id to pin at the start of the layout.
    :vartype start_state_id: int or None
    :ivar end_state_id: Optional state id to pin at the end of the layout.
    :vartype end_state_id: int or None
    """

    enabled: bool = Field(default=False)
    rankdir: str = Field(default="LR", min_length=1)
    min_edge_weight: float = Field(default=0.0, ge=0.0)
    start_state_id: Optional[int] = None
    end_state_id: Optional[int] = None


class MarkovAnalysisArtifactsConfig(AnalysisSchemaModel):
    """
    Artifact configuration for Markov analysis.

    :ivar graphviz: GraphViz export settings.
    :vartype graphviz: MarkovAnalysisArtifactsGraphVizConfig
    """

    graphviz: MarkovAnalysisArtifactsGraphVizConfig = Field(
        default_factory=MarkovAnalysisArtifactsGraphVizConfig
    )


class MarkovAnalysisReportConfig(AnalysisSchemaModel):
    """
    Report configuration for Markov analysis.

    :ivar max_state_exemplars: Maximum exemplar segments stored per state.
    :vartype max_state_exemplars: int
    :ivar state_naming: Optional provider-backed state naming configuration.
    :vartype state_naming: MarkovAnalysisStateNamingConfig or None
    """

    max_state_exemplars: int = Field(default=5, ge=0)
    state_naming: Optional["MarkovAnalysisStateNamingConfig"] = None


class MarkovAnalysisStateNamingConfig(AnalysisSchemaModel):
    """
    Provider-backed configuration for naming Markov states.

    :ivar enabled: Whether state naming is enabled.
    :vartype enabled: bool
    :ivar client: LLM client configuration.
    :vartype client: biblicus.ai.models.LlmClientConfig
    :ivar system_prompt: System prompt containing the context pack placeholder.
    :vartype system_prompt: str
    :ivar prompt_template: User prompt template for naming.
    :vartype prompt_template: str
    :ivar token_budget: Maximum tokens for the context pack text.
    :vartype token_budget: int
    :ivar max_exemplars_per_state: Maximum exemplars per state in the context pack.
    :vartype max_exemplars_per_state: int
    :ivar max_name_words: Maximum words allowed in each state name (short noun phrase).
    :vartype max_name_words: int
    :ivar max_retries: Maximum retries when the naming response is invalid.
    :vartype max_retries: int
    """

    enabled: bool = False
    client: Optional[LlmClientConfig] = None
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    token_budget: int = Field(default=256, ge=1)
    max_exemplars_per_state: int = Field(default=3, ge=1)
    max_name_words: int = Field(default=4, ge=1)
    max_retries: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _validate_state_naming(self) -> "MarkovAnalysisStateNamingConfig":
        if not self.enabled:
            return self
        if self.client is None:
            raise ValueError("report.state_naming.client is required when enabled")
        if self.system_prompt is None or not str(self.system_prompt).strip():
            raise ValueError("report.state_naming.system_prompt is required when enabled")
        if "{context_pack}" not in self.system_prompt:
            raise ValueError(
                'report.state_naming.system_prompt must include the "{context_pack}" placeholder'
            )
        if self.prompt_template is None or not str(self.prompt_template).strip():
            raise ValueError("report.state_naming.prompt_template is required when enabled")
        if "{context_pack}" in self.prompt_template:
            raise ValueError(
                'report.state_naming.prompt_template must not include "{context_pack}"'
            )
        return self


class MarkovAnalysisConfiguration(AnalysisSchemaModel):
    """
    Configuration for Markov analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar text_source: Text source configuration.
    :vartype text_source: MarkovAnalysisTextSourceConfig
    :ivar segmentation: Segmentation configuration.
    :vartype segmentation: MarkovAnalysisSegmentationConfig
    :ivar observations: Observation encoder configuration.
    :vartype observations: MarkovAnalysisObservationsConfig
    :ivar model: Markov model configuration.
    :vartype model: MarkovAnalysisModelConfig
    :ivar topic_modeling: Topic modeling configuration.
    :vartype topic_modeling: MarkovAnalysisTopicModelingConfig
    :ivar artifacts: Artifact configuration.
    :vartype artifacts: MarkovAnalysisArtifactsConfig
    :ivar report: Report configuration.
    :vartype report: MarkovAnalysisReportConfig
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    text_source: MarkovAnalysisTextSourceConfig = Field(
        default_factory=MarkovAnalysisTextSourceConfig
    )
    segmentation: MarkovAnalysisSegmentationConfig = Field(
        default_factory=MarkovAnalysisSegmentationConfig
    )
    observations: MarkovAnalysisObservationsConfig = Field(
        default_factory=MarkovAnalysisObservationsConfig
    )
    model: MarkovAnalysisModelConfig = Field(default_factory=MarkovAnalysisModelConfig)
    topic_modeling: MarkovAnalysisTopicModelingConfig = Field(
        default_factory=MarkovAnalysisTopicModelingConfig
    )
    llm_observations: MarkovAnalysisLlmObservationsConfig = Field(
        default_factory=MarkovAnalysisLlmObservationsConfig
    )
    embeddings: MarkovAnalysisEmbeddingsConfig = Field(
        default_factory=MarkovAnalysisEmbeddingsConfig
    )
    artifacts: MarkovAnalysisArtifactsConfig = Field(default_factory=MarkovAnalysisArtifactsConfig)
    report: MarkovAnalysisReportConfig = Field(default_factory=MarkovAnalysisReportConfig)

    @model_validator(mode="after")
    def _validate_schema_version(self) -> "MarkovAnalysisConfiguration":
        if self.schema_version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported analysis schema version: {self.schema_version}")
        return self


class MarkovAnalysisTextCollectionReport(AnalysisSchemaModel):
    """
    Report for Markov analysis text collection stage.

    :ivar status: Stage status.
    :vartype status: MarkovAnalysisStageStatus
    :ivar source_items: Count of items in extraction snapshot.
    :vartype source_items: int
    :ivar documents: Count of documents included.
    :vartype documents: int
    :ivar sample_size: Sample size applied.
    :vartype sample_size: int or None
    :ivar min_text_characters: Minimum length filter applied.
    :vartype min_text_characters: int or None
    :ivar empty_texts: Count of empty extracted texts.
    :vartype empty_texts: int
    :ivar skipped_items: Count of items skipped for missing/invalid text.
    :vartype skipped_items: int
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    status: MarkovAnalysisStageStatus
    source_items: int = Field(ge=0)
    documents: int = Field(ge=0)
    sample_size: Optional[int] = None
    min_text_characters: Optional[int] = None
    empty_texts: int = Field(ge=0)
    skipped_items: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class MarkovAnalysisSegment(AnalysisSchemaModel):
    """
    Segment record for Markov analysis.

    :ivar item_id: Source item identifier.
    :vartype item_id: str
    :ivar segment_index: One-based segment index within the item.
    :vartype segment_index: int
    :ivar text: Segment text.
    :vartype text: str
    """

    item_id: str = Field(min_length=1)
    segment_index: int = Field(ge=1)
    text: str = Field(min_length=1)


class MarkovAnalysisObservation(AnalysisSchemaModel):
    """
    Observation record for a single segment.

    :ivar item_id: Source item identifier.
    :vartype item_id: str
    :ivar segment_index: One-based segment index within the item.
    :vartype segment_index: int
    :ivar segment_text: Segment text.
    :vartype segment_text: str
    :ivar llm_label: Optional provider-proposed label.
    :vartype llm_label: str or None
    :ivar llm_label_confidence: Optional provider-proposed confidence.
    :vartype llm_label_confidence: float or None
    :ivar llm_summary: Optional provider-proposed summary.
    :vartype llm_summary: str or None
    :ivar topic_id: Optional topic identifier from topic modeling.
    :vartype topic_id: int or None
    :ivar topic_label: Optional topic label from topic modeling.
    :vartype topic_label: str or None
    :ivar embedding: Optional embedding vector for the configured embedding text source.
    :vartype embedding: list[float] or None
    :ivar token_count: Optional token count for the segment.
    :vartype token_count: int or None
    """

    item_id: str = Field(min_length=1)
    segment_index: int = Field(ge=1)
    segment_text: str = Field(min_length=1)
    llm_label: Optional[str] = None
    llm_label_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    llm_summary: Optional[str] = None
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = Field(default=None, ge=0)


class MarkovAnalysisState(AnalysisSchemaModel):
    """
    State record for Markov analysis.

    :ivar state_id: State identifier.
    :vartype state_id: int
    :ivar label: Optional human-readable label.
    :vartype label: str or None
    :ivar exemplars: Example segments representative of the state.
    :vartype exemplars: list[str]
    """

    state_id: int = Field(ge=0)
    label: Optional[str] = None
    exemplars: List[str] = Field(default_factory=list)


class MarkovAnalysisTransition(AnalysisSchemaModel):
    """
    Directed transition edge between two states.

    :ivar from_state: Source state identifier.
    :vartype from_state: int
    :ivar to_state: Destination state identifier.
    :vartype to_state: int
    :ivar weight: Transition weight.
    :vartype weight: float
    """

    from_state: int = Field(ge=0)
    to_state: int = Field(ge=0)
    weight: float


class MarkovAnalysisDecodedPath(AnalysisSchemaModel):
    """
    Decoded state sequence for a single item.

    :ivar item_id: Source item identifier.
    :vartype item_id: str
    :ivar state_sequence: Most likely state sequence over segments.
    :vartype state_sequence: list[int]
    """

    item_id: str = Field(min_length=1)
    state_sequence: List[int] = Field(default_factory=list)


class MarkovAnalysisReport(AnalysisSchemaModel):
    """
    Markov analysis report data.

    :ivar text_collection: Text collection report.
    :vartype text_collection: MarkovAnalysisTextCollectionReport
    :ivar status: Overall analysis status.
    :vartype status: MarkovAnalysisStageStatus
    :ivar states: State records.
    :vartype states: list[MarkovAnalysisState]
    :ivar transitions: Transition edges.
    :vartype transitions: list[MarkovAnalysisTransition]
    :ivar decoded_paths: Per-item decoded paths.
    :vartype decoded_paths: list[MarkovAnalysisDecodedPath]
    :ivar topic_modeling: Optional topic modeling report for segment topics.
    :vartype topic_modeling: TopicModelingReport or None
    :ivar warnings: Warning messages.
    :vartype warnings: list[str]
    :ivar errors: Error messages.
    :vartype errors: list[str]
    """

    text_collection: MarkovAnalysisTextCollectionReport
    status: MarkovAnalysisStageStatus
    states: List[MarkovAnalysisState] = Field(default_factory=list)
    transitions: List[MarkovAnalysisTransition] = Field(default_factory=list)
    decoded_paths: List[MarkovAnalysisDecodedPath] = Field(default_factory=list)
    topic_modeling: Optional[TopicModelingReport] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class MarkovAnalysisOutput(AnalysisSchemaModel):
    """
    Output bundle for Markov analysis.

    :ivar schema_version: Analysis schema version.
    :vartype schema_version: int
    :ivar analysis_id: Analysis backend identifier.
    :vartype analysis_id: str
    :ivar generated_at: International Organization for Standardization 8601 timestamp for output creation.
    :vartype generated_at: str
    :ivar snapshot: Analysis snapshot manifest.
    :vartype snapshot: AnalysisRunManifest
    :ivar report: Markov analysis report data.
    :vartype report: MarkovAnalysisReport
    """

    schema_version: int = Field(default=ANALYSIS_SCHEMA_VERSION, ge=1)
    analysis_id: str
    generated_at: str
    snapshot: AnalysisRunManifest
    report: MarkovAnalysisReport
