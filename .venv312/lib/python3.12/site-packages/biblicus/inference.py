"""
Inference backend abstraction for machine learning powered components.

This module provides reusable configuration and credential resolution patterns for components
that can execute locally or via API providers.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceBackendMode(str, Enum):
    """Execution mode for inference backends."""

    LOCAL = "local"
    API = "api"


class ApiProvider(str, Enum):
    """Supported application programming interface providers for inference."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class InferenceBackendConfig(BaseModel):
    """
    Composable configuration for inference backends.

    This config can be embedded in extractor or transformer configurations to provide
    a uniform interface for local versus application programming interface execution.

    :ivar mode: Execution mode, local or application programming interface.
    :vartype mode: InferenceBackendMode
    :ivar api_provider: Application programming interface provider when mode is application programming interface.
    :vartype api_provider: ApiProvider or None
    :ivar api_key: Optional per-config application programming interface key override.
    :vartype api_key: str or None
    :ivar model_id: Optional model identifier for application programming interface requests.
    :vartype model_id: str or None
    """

    model_config = ConfigDict(extra="forbid")

    mode: InferenceBackendMode = Field(default=InferenceBackendMode.LOCAL)
    api_provider: Optional[ApiProvider] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _validate_api_provider_required(self) -> "InferenceBackendConfig":
        if self.mode == InferenceBackendMode.API and self.api_provider is None:
            raise ValueError("api_provider is required when mode is 'api'")
        return self


def resolve_api_key(
    provider: ApiProvider,
    *,
    config_override: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve an application programming interface key with precedence rules.

    Precedence order (highest to lowest):
    1. Explicit config override parameter
    2. Environment variable for the provider
    3. User configuration file

    :param provider: Application programming interface provider to resolve key for.
    :type provider: ApiProvider
    :param config_override: Optional explicit key from configuration.
    :type config_override: str or None
    :return: Resolved application programming interface key or None if unavailable.
    :rtype: str or None
    """
    if config_override is not None:
        return config_override

    from .user_config import load_user_config

    if provider == ApiProvider.HUGGINGFACE:
        env_key = os.environ.get("HUGGINGFACE_API_KEY")
        if env_key:
            return env_key
        user_config = load_user_config()
        if user_config.huggingface is not None:
            return user_config.huggingface.api_key
        return None
    elif provider == ApiProvider.OPENAI:
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key
        user_config = load_user_config()
        if user_config.openai is not None:
            return user_config.openai.api_key
        return None
    else:
        return None
