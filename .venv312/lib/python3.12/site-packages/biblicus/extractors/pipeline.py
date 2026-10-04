"""
Pipeline extractor configuration and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import ExtractionSnapshotFatalError
from ..models import CatalogItem, ExtractedText, ExtractionStageOutput
from .base import TextExtractor

Corpus = Any


class PipelineStageSpec(BaseModel):
    """
    Single extractor stage within a pipeline.

    :ivar extractor_id: Extractor plugin identifier.
    :vartype extractor_id: str
    :ivar configuration: Extractor configuration mapping.
    :vartype configuration: dict[str, Any]
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    extractor_id: str = Field(min_length=1)
    configuration: Dict[str, Any] = Field(default_factory=dict, alias="config")


class PipelineExtractorConfig(BaseModel):
    """
    Configuration for the pipeline extractor.

    :ivar stages: Ordered list of extractor stages to run.
    :vartype stages: list[PipelineStageSpec]
    """

    model_config = ConfigDict(extra="forbid")

    stages: List[PipelineStageSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _forbid_pipeline_stage(self) -> "PipelineExtractorConfig":
        if any(stage.extractor_id == "pipeline" for stage in self.stages):
            raise ValueError("Pipeline stages cannot include the pipeline extractor itself")
        return self


class PipelineExtractor(TextExtractor):
    """
    Pipeline extractor configuration shim.

    The pipeline extractor is executed by the extraction engine so it can persist
    per-stage artifacts. This class only validates configuration.

    :ivar extractor_id: Extractor identifier.
    :vartype extractor_id: str
    """

    extractor_id = "pipeline"

    def validate_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Validate pipeline configuration.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :return: Parsed configuration.
        :rtype: PipelineExtractorConfig
        """
        return PipelineExtractorConfig.model_validate(config)

    def extract_text(
        self,
        *,
        corpus: "Corpus",
        item: CatalogItem,
        config: BaseModel,
        previous_extractions: List[ExtractionStageOutput],
    ) -> Optional[ExtractedText]:
        """
        Reject direct execution of the pipeline extractor.

        :param corpus: Corpus containing the item bytes.
        :type corpus: Corpus
        :param item: Catalog item being processed.
        :type item: CatalogItem
        :param config: Parsed configuration model.
        :type config: PipelineExtractorConfig
        :param previous_extractions: Prior stage outputs for this item within the pipeline.
        :type previous_extractions: list[biblicus.models.ExtractionStageOutput]
        :raises ExtractionSnapshotFatalError: Always, because the pipeline is executed by the runner.
        :return: None.
        :rtype: None
        """
        _ = corpus
        _ = item
        _ = config
        _ = previous_extractions
        raise ExtractionSnapshotFatalError(
            "Pipeline extractor must be executed by the extraction snapshotner."
        )
