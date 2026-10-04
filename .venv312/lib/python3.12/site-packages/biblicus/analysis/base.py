"""
Analysis backend interface for Biblicus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from pydantic import BaseModel

from ..corpus import Corpus
from ..models import ExtractionSnapshotReference


class CorpusAnalysisBackend(ABC):
    """
    Abstract interface for analysis backends.

    :ivar analysis_id: Identifier string for the analysis backend.
    :vartype analysis_id: str
    """

    analysis_id: str

    @abstractmethod
    def run_analysis(
        self,
        corpus: Corpus,
        *,
        configuration_name: str,
        configuration: Dict[str, object],
        extraction_snapshot: ExtractionSnapshotReference,
    ) -> BaseModel:
        """
        Run an analysis pipeline for a corpus.

        :param corpus: Corpus to analyze.
        :type corpus: Corpus
        :param configuration_name: Human-readable configuration name.
        :type configuration_name: str
        :param configuration: Analysis configuration values.
        :type configuration: dict[str, object]
        :param extraction_snapshot: Extraction snapshot reference for text inputs.
        :type extraction_snapshot: biblicus.models.ExtractionSnapshotReference
        :return: Analysis output model.
        :rtype: pydantic.BaseModel
        """
        raise NotImplementedError
