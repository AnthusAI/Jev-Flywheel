"""
User configuration file loading for Biblicus.

User configuration is intended for small, local settings such as credentials for optional
integrations. It is separate from corpus configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._vendor.dotyaml import ConfigLoader


class OpenAiUserConfig(BaseModel):
    """
    Configuration for OpenAI integrations.

    :ivar api_key: OpenAI API key used for authenticated requests.
    :vartype api_key: str
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)


class HuggingFaceUserConfig(BaseModel):
    """
    Configuration for HuggingFace integrations.

    :ivar api_key: HuggingFace API key used for authenticated requests.
    :vartype api_key: str
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)


class DeepgramUserConfig(BaseModel):
    """
    Configuration for Deepgram integrations.

    :ivar api_key: Deepgram API key used for authenticated requests.
    :vartype api_key: str
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)


class AldeaUserConfig(BaseModel):
    """
    Configuration for Aldea integrations.

    :ivar api_key: Aldea API key used for authenticated requests.
    :vartype api_key: str
    """

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)


class Neo4jUserConfig(BaseModel):
    """
    Configuration for Neo4j integrations.

    :ivar uri: Neo4j connection URI.
    :vartype uri: str
    :ivar username: Neo4j username.
    :vartype username: str
    :ivar password: Neo4j password.
    :vartype password: str
    :ivar database: Optional Neo4j database name.
    :vartype database: str
    :ivar auto_start: Whether to auto-start Neo4j via Docker.
    :vartype auto_start: bool
    :ivar container_name: Docker container name for Neo4j.
    :vartype container_name: str
    :ivar docker_image: Docker image reference for Neo4j.
    :vartype docker_image: str
    :ivar http_port: HTTP port for the Neo4j container.
    :vartype http_port: int
    :ivar bolt_port: Bolt port for the Neo4j container.
    :vartype bolt_port: int
    """

    model_config = ConfigDict(extra="forbid")

    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    auto_start: bool = True
    container_name: str = Field(default="biblicus-neo4j", min_length=1)
    docker_image: str = Field(default="neo4j:5", min_length=1)
    http_port: int = Field(default=7474, ge=1)
    bolt_port: int = Field(default=7687, ge=1)


class SourceProfileConfig(BaseModel):
    """
    Configuration for remote source profiles.

    :ivar name: Unique profile name.
    :vartype name: str
    :ivar kind: Remote source kind (s3 or azure-blob).
    :vartype kind: str
    :ivar access_key_id: AWS access key identifier.
    :vartype access_key_id: str or None
    :ivar secret_access_key: AWS secret access key.
    :vartype secret_access_key: str or None
    :ivar session_token: Optional AWS session token.
    :vartype session_token: str or None
    :ivar region: Optional AWS region.
    :vartype region: str or None
    :ivar endpoint_url: Optional S3-compatible endpoint URL.
    :vartype endpoint_url: str or None
    :ivar connection_string: Azure Storage connection string.
    :vartype connection_string: str or None
    :ivar account_name: Optional Azure storage account name.
    :vartype account_name: str or None
    :ivar account_key: Optional Azure storage account key.
    :vartype account_key: str or None
    :ivar account_url: Optional Azure storage account URL.
    :vartype account_url: str or None
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    account_url: Optional[str] = None

    @model_validator(mode="after")
    def _validate_kind(self) -> "SourceProfileConfig":
        if self.kind not in {"s3", "azure-blob"}:
            raise ValueError(f"Unsupported source profile kind: {self.kind}")
        return self


class AzureStorageUserConfig(BaseModel):
    """
    Azure Storage account configuration.

    :ivar connection_string: Optional connection string for blob access.
    :vartype connection_string: str or None
    :ivar account_name: Optional storage account name.
    :vartype account_name: str or None
    :ivar account_key: Optional storage account key.
    :vartype account_key: str or None
    """

    model_config = ConfigDict(extra="allow")

    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None


class BiblicusUserConfig(BaseModel):
    """
    Parsed user configuration for Biblicus.

    :ivar openai: Optional OpenAI configuration.
    :vartype openai: OpenAiUserConfig or None
    :ivar huggingface: Optional HuggingFace configuration.
    :vartype huggingface: HuggingFaceUserConfig or None
    :ivar deepgram: Optional Deepgram configuration.
    :vartype deepgram: DeepgramUserConfig or None
    :ivar aldea: Optional Aldea configuration.
    :vartype aldea: AldeaUserConfig or None
    :ivar neo4j: Optional Neo4j configuration.
    :vartype neo4j: Neo4jUserConfig or None
    :ivar sources: Optional remote source profiles.
    :vartype sources: list[SourceProfileConfig] or None
    """

    model_config = ConfigDict(extra="forbid")

    openai: Optional[OpenAiUserConfig] = None
    huggingface: Optional[HuggingFaceUserConfig] = None
    deepgram: Optional[DeepgramUserConfig] = None
    aldea: Optional[AldeaUserConfig] = None
    neo4j: Optional[Neo4jUserConfig] = None
    sources: Optional[list[SourceProfileConfig]] = None
    azure_storage: Optional[AzureStorageUserConfig] = None


def default_user_config_paths(
    *, cwd: Optional[Path] = None, home: Optional[Path] = None
) -> list[Path]:
    """
    Compute the default user configuration file search paths.

    The search order is:

    1. Home configuration: ``~/.biblicus/config.yml``
    2. Local configuration: ``./.biblicus/config.yml``

    Local configuration overrides home configuration when both exist.

    :param cwd: Optional working directory to use instead of the process current directory.
    :type cwd: Path or None
    :param home: Optional home directory to use instead of the current user's home directory.
    :type home: Path or None
    :return: Ordered list of configuration file paths.
    :rtype: list[Path]
    """
    resolved_home = (home or Path.home()).expanduser()
    resolved_cwd = cwd or Path.cwd()
    return [
        resolved_home / ".biblicus" / "config.yml",
        resolved_cwd / ".biblicus" / "config.yml",
    ]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_dotyaml_data(path: Path) -> Dict[str, Any]:
    """
    Load a dotyaml configuration file and return a nested mapping.

    :param path: Configuration file path.
    :type path: Path
    :return: Parsed YAML data mapping.
    :rtype: dict[str, Any]
    """
    loader = ConfigLoader(prefix="", load_dotenv_first=False)
    loaded = loader.load_from_yaml(path)
    return loaded if isinstance(loaded, dict) else {}


def load_user_config(*, paths: Optional[list[Path]] = None) -> BiblicusUserConfig:
    """
    Load user configuration from known locations.

    This function merges multiple configuration files in order. Later files override earlier files.

    :param paths: Optional explicit search paths. When omitted, the default paths are used.
    :type paths: list[Path] or None
    :return: Parsed user configuration. When no files exist, the configuration is empty.
    :rtype: BiblicusUserConfig
    :raises ValueError: If an existing configuration file is not parseable.
    """
    search_paths = paths or default_user_config_paths()
    merged_data: Dict[str, Any] = {}

    for path in search_paths:
        if not path.is_file():
            continue
        loaded = _load_dotyaml_data(path)
        merged_data = _deep_merge(merged_data, loaded)

    return BiblicusUserConfig.model_validate(merged_data)


def resolve_openai_api_key(*, config: Optional[BiblicusUserConfig] = None) -> Optional[str]:
    """
    Resolve an OpenAI API key from environment or user configuration.

    Environment takes precedence over configuration.

    :param config: Optional pre-loaded user configuration.
    :type config: BiblicusUserConfig or None
    :return: API key string, or None when no key is available.
    :rtype: str or None
    """
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    loaded = config or load_user_config()
    if loaded.openai is None:
        return None
    return loaded.openai.api_key


def resolve_huggingface_api_key(*, config: Optional[BiblicusUserConfig] = None) -> Optional[str]:
    """
    Resolve a HuggingFace API key from environment or user configuration.

    Environment takes precedence over configuration.

    :param config: Optional pre-loaded user configuration.
    :type config: BiblicusUserConfig or None
    :return: API key string, or None when no key is available.
    :rtype: str or None
    """
    env_key = os.environ.get("HUGGINGFACE_API_KEY")
    if env_key:
        return env_key
    loaded = config or load_user_config()
    if loaded.huggingface is None:
        return None
    return loaded.huggingface.api_key


def resolve_deepgram_api_key(*, config: Optional[BiblicusUserConfig] = None) -> Optional[str]:
    """
    Resolve a Deepgram API key from environment or user configuration.

    Environment takes precedence over configuration.

    :param config: Optional pre-loaded user configuration.
    :type config: BiblicusUserConfig or None
    :return: API key string, or None when no key is available.
    :rtype: str or None
    """
    env_key = os.environ.get("DEEPGRAM_API_KEY")
    if env_key:
        return env_key
    loaded = config or load_user_config()
    if loaded.deepgram is None:
        return None
    return loaded.deepgram.api_key


def resolve_aldea_api_key(*, config: Optional[BiblicusUserConfig] = None) -> Optional[str]:
    """
    Resolve an Aldea API key from environment or user configuration.

    Environment takes precedence over configuration.

    :param config: Optional pre-loaded user configuration.
    :type config: BiblicusUserConfig or None
    :return: API key string, or None when no key is available.
    :rtype: str or None
    """
    env_key = os.environ.get("ALDEA_API_KEY")
    if env_key:
        return env_key
    loaded = config or load_user_config()
    if loaded.aldea is None:
        return None
    return loaded.aldea.api_key


def resolve_source_profile(
    profile_name: str, *, config: Optional[BiblicusUserConfig] = None
) -> SourceProfileConfig:
    """
    Resolve a remote source profile from user configuration and environment.

    Environment takes precedence over configuration.

    :param profile_name: Profile name to resolve.
    :type profile_name: str
    :param config: Optional pre-loaded user configuration.
    :type config: BiblicusUserConfig or None
    :return: Resolved source profile configuration.
    :rtype: SourceProfileConfig
    :raises ValueError: If the profile is missing or incomplete.
    """
    if not profile_name or not str(profile_name).strip():
        raise ValueError("Source profile name must be provided")
    loaded = config or load_user_config()
    profiles = list(loaded.sources or [])
    profile = next((item for item in profiles if item.name == profile_name), None)
    if profile is None:
        raise ValueError(f"Source profile not found: {profile_name}")

    resolved = profile.model_copy()

    if resolved.kind == "s3":
        env_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        env_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        env_session_token = os.environ.get("AWS_SESSION_TOKEN")
        env_region = os.environ.get("AWS_REGION")
        resolved.access_key_id = env_access_key or resolved.access_key_id
        resolved.secret_access_key = env_secret_key or resolved.secret_access_key
        resolved.session_token = env_session_token or resolved.session_token
        resolved.region = env_region or resolved.region
        if not (resolved.access_key_id and resolved.secret_access_key):
            raise ValueError(
                "S3 credentials not found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or configure sources in .biblicus/config.yml."
            )
    elif resolved.kind == "azure-blob":
        env_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        env_account = os.environ.get("AZURE_STORAGE_ACCOUNT")
        env_key = os.environ.get("AZURE_STORAGE_KEY")
        resolved.connection_string = env_connection_string or resolved.connection_string
        resolved.account_name = env_account or resolved.account_name
        resolved.account_key = env_key or resolved.account_key
        if resolved.connection_string and not resolved.account_name:
            parsed_account = _parse_account_name_from_connection_string(
                resolved.connection_string
            )
            resolved.account_name = parsed_account or resolved.account_name
        if resolved.connection_string:
            return resolved
        if not (resolved.account_name and resolved.account_key):
            raise ValueError(
                "Azure storage credentials not found. Set AZURE_STORAGE_CONNECTION_STRING "
                "or configure sources in .biblicus/config.yml."
            )
    else:
        raise ValueError(f"Unsupported source profile kind: {resolved.kind}")

    return resolved


def _parse_account_name_from_connection_string(connection_string: str) -> Optional[str]:
    for part in connection_string.split(";"):
        if part.startswith("AccountName="):
            value = part.split("=", 1)[1].strip()
            if value:
                return value
    return None
