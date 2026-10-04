"""
Core loading functionality for dotyaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .interpolation import interpolate_env_vars
from .transformer import flatten_dict, unflatten_env_vars


def load_config(
    yaml_path: Optional[Union[str, Path]] = None,
    prefix: str = "",
    override: bool = False,
    dotenv_path: Optional[Union[str, Path]] = ".env",
    load_dotenv_first: bool = True,
) -> Dict[str, str]:
    """
    Load configuration from a YAML file and set environment variables.

    :param yaml_path: Path to YAML configuration file. When None, only reads existing env vars.
    :type yaml_path: str or Path or None
    :param prefix: Prefix for environment variable names (for example, ``APP``).
    :type prefix: str
    :param override: Whether to override existing environment variables.
    :type override: bool
    :param dotenv_path: Optional ``.env`` file path to load first.
    :type dotenv_path: str or Path or None
    :param load_dotenv_first: Whether to load ``.env`` before YAML.
    :type load_dotenv_first: bool
    :return: Mapping of values that were set.
    :rtype: dict[str, str]
    """

    config: Dict[str, str] = {}

    if load_dotenv_first and DOTENV_AVAILABLE and dotenv_path:
        env_file = Path(dotenv_path)
        env_locations: list[Path] = []

        if env_file.is_absolute():
            env_locations.append(env_file)
        else:
            env_locations.append(Path.cwd() / dotenv_path)
            if yaml_path:
                yaml_dir = Path(yaml_path).parent
                env_locations.append(yaml_dir / dotenv_path)

        for env_path in env_locations:
            if env_path.exists():
                load_dotenv(env_path)
                break

    if yaml_path and Path(yaml_path).exists():
        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if yaml_data:
            yaml_data = interpolate_env_vars(yaml_data)
            flat_config = flatten_dict(yaml_data, prefix)

            for key, value in flat_config.items():
                if not override and key in os.environ:
                    config[key] = os.environ[key]
                else:
                    os.environ[key] = value
                    config[key] = value

    return config


def _merge_mapping_values(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping_values(existing, value)
        else:
            merged[key] = value
    return merged


def load_yaml_view(paths: Iterable[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load and compose one or more YAML files into a single mapping.

    :param paths: Iterable of YAML file paths in precedence order.
    :type paths: Iterable[str | Path]
    :return: Composed YAML mapping.
    :rtype: dict[str, Any]
    :raises ValueError: If any YAML file does not contain a mapping.
    """

    composed: Dict[str, Any] = {}
    for raw_path in paths:
        yaml_path = Path(raw_path)
        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if yaml_data is None:
            yaml_data = {}

        if not isinstance(yaml_data, dict):
            raise ValueError(f"YAML content must be a mapping for {yaml_path}")

        yaml_data = interpolate_env_vars(yaml_data)
        composed = _merge_mapping_values(composed, yaml_data)
    return composed


class ConfigLoader:
    """
    Configuration loader that can read YAML files or environment variables.
    """

    def __init__(
        self,
        prefix: str = "",
        schema: Optional[Dict[str, Any]] = None,
        dotenv_path: Optional[Union[str, Path]] = ".env",
        load_dotenv_first: bool = True,
    ):
        self.prefix = prefix
        self.schema = schema
        self.dotenv_path = dotenv_path
        self.load_dotenv_first = load_dotenv_first

        if self.load_dotenv_first and DOTENV_AVAILABLE and self.dotenv_path:
            env_file = Path(self.dotenv_path)
            env_locations: list[Path] = []

            if env_file.is_absolute():
                env_locations.append(env_file)
            else:
                env_locations.append(Path.cwd() / self.dotenv_path)

            for env_path in env_locations:
                if env_path.exists():
                    load_dotenv(env_path)
                    break

    def load_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file with environment variable interpolation.

        :param yaml_path: YAML configuration file path.
        :type yaml_path: str or Path
        :return: Parsed YAML data.
        :rtype: dict[str, Any]
        """

        if not Path(yaml_path).exists():
            return {}

        if self.load_dotenv_first and DOTENV_AVAILABLE and self.dotenv_path:
            env_file = Path(self.dotenv_path)
            env_locations: list[Path] = []

            if env_file.is_absolute():
                env_locations.append(env_file)
            else:
                env_locations.append(Path.cwd() / self.dotenv_path)
                yaml_dir = Path(yaml_path).parent
                env_locations.append(yaml_dir / self.dotenv_path)

            for env_path in env_locations:
                if env_path.exists():
                    load_dotenv(env_path)
                    break

        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if yaml_data:
            yaml_data = interpolate_env_vars(yaml_data)

        return yaml_data or {}

    def load_from_env(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        :return: Nested configuration dictionary.
        :rtype: dict[str, Any]
        """

        env_vars = dict(os.environ)
        return unflatten_env_vars(env_vars, self.prefix)

    def set_env_vars(self, config: Dict[str, Any], override: bool = False) -> None:
        """
        Set environment variables from a configuration dictionary.

        :param config: Configuration mapping.
        :type config: dict[str, Any]
        :param override: Whether to override existing environment variables.
        :type override: bool
        :return: None.
        :rtype: None
        """

        flat_config = flatten_dict(config, self.prefix)

        for key, value in flat_config.items():
            if override or key not in os.environ:
                os.environ[key] = value
