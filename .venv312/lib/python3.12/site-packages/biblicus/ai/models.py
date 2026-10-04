"""
Pydantic models for provider-backed AI clients.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field, field_validator

from ..analysis.schema import AnalysisSchemaModel
from ..user_config import resolve_openai_api_key


class AiProvider(str, Enum):
    """
    Supported AI providers.
    """

    OPENAI = "openai"
    BEDROCK = "bedrock"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LITELLM = "litellm"


def _normalize_provider(value: object, *, error_label: str) -> str:
    if isinstance(value, AiProvider):
        return value.value
    if isinstance(value, str):
        return value.lower()
    raise ValueError(f"{error_label} must be a string or AiProvider")


def _litellm_model(provider: str, model: str) -> str:
    normalized_model = model.strip()
    if "/" in normalized_model:
        return normalized_model
    return f"{provider}/{normalized_model}"


class LlmClientConfig(AnalysisSchemaModel):
    """
    Configuration for a chat completion invocation.

    :ivar provider: Provider identifier.
    :vartype provider: str or AiProvider
    :ivar model: Model identifier.
    :vartype model: str
    :ivar api_key: Optional API key override.
    :vartype api_key: str or None
    :ivar api_base: Optional API base override.
    :vartype api_base: str or None
    :ivar temperature: Optional generation temperature.
    :vartype temperature: float or None
    :ivar max_tokens: Optional maximum output tokens.
    :vartype max_tokens: int or None
    :ivar response_format: Optional response format identifier.
    :vartype response_format: str or None
    :ivar max_retries: Optional maximum retry count for transient failures.
    :vartype max_retries: int
    :ivar timeout_seconds: Optional request timeout in seconds.
    :vartype timeout_seconds: float or None
    :ivar model_type: Optional model type identifier.
    :vartype model_type: str or None
    :ivar extra_params: Additional provider-specific parameters to pass through.
    :vartype extra_params: dict[str, Any]
    """

    provider: str
    model: str = Field(min_length=1)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    response_format: Optional[str] = None
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: Optional[float] = Field(default=None, gt=0.0)
    model_type: Optional[str] = None
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _parse_provider(cls, value: object) -> str:
        return _normalize_provider(value, error_label="llm client provider")

    def litellm_model(self) -> str:
        """
        Resolve the DSPy model identifier for this client.

        :return: DSPy model string (LiteLLM format).
        :rtype: str
        """
        return _litellm_model(self.provider, self.model)

    def resolve_api_key(self) -> Optional[str]:
        """
        Resolve an API key for the configured provider.

        :return: API key string or None if not required.
        :rtype: str or None
        :raises ValueError: If OpenAI is configured and no key is available.
        """
        if self.api_key:
            return self.api_key
        if self.provider != AiProvider.OPENAI.value:
            return None
        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ValueError(
                "OpenAI provider requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )
        return api_key

    def build_litellm_kwargs(self) -> dict[str, Any]:
        """
        Build DSPy keyword arguments for chat completions.

        :return: Keyword arguments for DSPy (LiteLLM-backed).
        :rtype: dict[str, Any]
        """
        api_key = self.resolve_api_key()
        base_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_type": self.model_type,
            "timeout": self.timeout_seconds,
            "num_retries": self.max_retries,
        }
        for key, value in (self.extra_params or {}).items():
            base_kwargs[key] = value
        return {key: value for key, value in base_kwargs.items() if value is not None}


class EmbeddingsClientConfig(AnalysisSchemaModel):
    """
    Configuration for an embeddings invocation.

    :ivar provider: Provider identifier.
    :vartype provider: str or AiProvider
    :ivar model: Model identifier.
    :vartype model: str
    :ivar api_key: Optional API key override.
    :vartype api_key: str or None
    :ivar api_base: Optional API base override.
    :vartype api_base: str or None
    :ivar batch_size: Maximum number of texts per request.
    :vartype batch_size: int
    :ivar parallelism: Maximum number of concurrent requests.
    :vartype parallelism: int
    :ivar max_retries: Optional maximum retry count for transient failures.
    :vartype max_retries: int
    :ivar timeout_seconds: Optional request timeout in seconds.
    :vartype timeout_seconds: float or None
    :ivar extra_params: Additional provider-specific parameters to pass through.
    :vartype extra_params: dict[str, Any]
    """

    provider: str
    model: str = Field(min_length=1)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    batch_size: int = Field(default=64, ge=1)
    parallelism: int = Field(default=4, ge=1)
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: Optional[float] = Field(default=None, gt=0.0)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _parse_provider(cls, value: object) -> str:
        return _normalize_provider(value, error_label="embeddings provider")

    def litellm_model(self) -> str:
        """
        Resolve the DSPy model identifier for this client.

        :return: DSPy model string (LiteLLM format).
        :rtype: str
        """
        return _litellm_model(self.provider, self.model)

    def resolve_api_key(self) -> Optional[str]:
        """
        Resolve an API key for the configured provider.

        :return: API key string or None if not required.
        :rtype: str or None
        :raises ValueError: If OpenAI is configured and no key is available.
        """
        if self.api_key:
            return self.api_key
        if self.provider != AiProvider.OPENAI.value:
            return None
        api_key = resolve_openai_api_key()
        if api_key is None:
            raise ValueError(
                "OpenAI provider requires an OpenAI API key. "
                "Set OPENAI_API_KEY or configure it in ~/.biblicus/config.yml or ./.biblicus/config.yml under "
                "openai.api_key."
            )
        return api_key

    def build_litellm_kwargs(self) -> dict[str, Any]:
        """
        Build DSPy keyword arguments for embeddings calls.

        :return: Keyword arguments for DSPy (LiteLLM-backed).
        :rtype: dict[str, Any]
        """
        api_key = self.resolve_api_key()
        base_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_base": self.api_base,
            "timeout": self.timeout_seconds,
            "num_retries": self.max_retries,
        }
        for key, value in (self.extra_params or {}).items():
            base_kwargs[key] = value
        return {key: value for key, value in base_kwargs.items() if value is not None}
