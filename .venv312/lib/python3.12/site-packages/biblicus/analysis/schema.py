"""
Shared schema utilities for analysis models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AnalysisSchemaModel(BaseModel):
    """
    Base model for analysis schemas with strict validation.

    :ivar model_config: Pydantic configuration for strict schema enforcement.
    :vartype model_config: pydantic.ConfigDict
    """

    model_config = ConfigDict(extra="forbid", strict=True)
