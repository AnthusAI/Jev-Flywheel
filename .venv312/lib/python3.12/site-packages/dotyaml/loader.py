"""
Core loading functionality for yamlenv
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union, Iterable, List, Tuple

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .transformer import flatten_dict, unflatten_env_vars
from .interpolation import interpolate_env_vars


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dictionaries.

    - Mapping values are merged recursively.
    - Non-mapping values replace earlier values.
    - Lists are treated as atomic values and replace earlier values.

    :param base: Base mapping.
    :type base: dict[str, Any]
    :param override: Override mapping applied on top of base.
    :type override: dict[str, Any]
    :return: New mapping with overrides applied.
    :rtype: dict[str, Any]
    """
    merged: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge_dicts(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def load_yaml(
    yaml_path: Union[str, Path],
    *,
    dotenv_path: Optional[Union[str, Path]] = ".env",
    load_dotenv_first: bool = True,
) -> Dict[str, Any]:
    """
    Load YAML configuration from a file with environment interpolation.

    This function does not set environment variables. It only returns the parsed mapping.

    :param yaml_path: Path to YAML configuration file.
    :type yaml_path: str or pathlib.Path
    :param dotenv_path: Optional .env file to load first for interpolation.
    :type dotenv_path: str or pathlib.Path or None
    :param load_dotenv_first: Whether to load dotenv before parsing YAML.
    :type load_dotenv_first: bool
    :return: Parsed YAML mapping.
    :rtype: dict[str, Any]
    :raises ValueError: If YAML content is not a mapping/object.
    """
    loader = ConfigLoader(dotenv_path=dotenv_path, load_dotenv_first=load_dotenv_first)
    data = loader.load_from_yaml(yaml_path)
    if not isinstance(data, dict):
        raise ValueError("YAML configuration must be a mapping/object")
    return data


def load_yaml_view(
    yaml_paths: Iterable[Union[str, Path]],
    *,
    dotenv_path: Optional[Union[str, Path]] = ".env",
    load_dotenv_first: bool = True,
) -> Dict[str, Any]:
    """
    Load and compose multiple YAML configuration files into a single configuration view.

    Files are merged in order. Later files override earlier files using :func:`deep_merge_dicts`.

    :param yaml_paths: Iterable of YAML file paths.
    :type yaml_paths: Iterable[str | pathlib.Path]
    :param dotenv_path: Optional .env file to load first for interpolation.
    :type dotenv_path: str or pathlib.Path or None
    :param load_dotenv_first: Whether to load dotenv before parsing YAML.
    :type load_dotenv_first: bool
    :return: Composed YAML mapping.
    :rtype: dict[str, Any]
    :raises FileNotFoundError: If any referenced file does not exist.
    :raises ValueError: If any YAML file is not a mapping/object.
    """
    paths: List[Path] = [Path(path) for path in yaml_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found: {path}")

    composed: Dict[str, Any] = {}
    for path in paths:
        loaded = load_yaml(
            path, dotenv_path=dotenv_path, load_dotenv_first=load_dotenv_first
        )
        composed = deep_merge_dicts(composed, loaded)
    return composed


def load_config(
    yaml_path: Optional[Union[str, Path]] = None,
    prefix: str = "",
    override: bool = False,
    dotenv_path: Optional[Union[str, Path]] = ".env",
    load_dotenv_first: bool = True,
) -> Dict[str, str]:
    """
    Load configuration from YAML file and set environment variables.

    Args:
        yaml_path: Path to YAML configuration file. If None, only reads existing env vars.
        prefix: Prefix for environment variable names (e.g., 'APP' -> 'APP_DATABASE_HOST')
        override: If True, override existing environment variables
        dotenv_path: Path to .env file to load first. Set to None to disable.
        load_dotenv_first: If True, automatically load .env file before processing YAML

    Returns:
        Dictionary of all configuration values that were set
    """
    config = {}

    # Load .env file first if available and requested
    if load_dotenv_first and DOTENV_AVAILABLE and dotenv_path:
        env_file = Path(dotenv_path)

        # Try multiple locations for .env file
        env_locations = []

        # 1. Exact path specified
        if env_file.is_absolute():
            env_locations.append(env_file)
        else:
            # 2. Current working directory
            env_locations.append(Path.cwd() / dotenv_path)

            # 3. Same directory as YAML file (if yaml_path provided)
            if yaml_path:
                yaml_dir = Path(yaml_path).parent
                env_locations.append(yaml_dir / dotenv_path)

        # Load from the first .env file found
        for env_path in env_locations:
            if env_path.exists():
                load_dotenv(env_path)
                break

    if yaml_path and Path(yaml_path).exists():
        # Load and parse YAML file
        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if yaml_data:
            # Interpolate environment variables in the YAML data
            interpolated_data = interpolate_env_vars(yaml_data)

            if isinstance(interpolated_data, dict):
                # Transform nested structure to flat env var names
                flat_config = flatten_dict(interpolated_data, prefix)

                # Set environment variables with precedence handling
                for key, value in flat_config.items():
                    # Check if env var already exists and override is False
                    if not override and key in os.environ:
                        # Keep existing env var, but track it in config
                        config[key] = os.environ[key]
                    else:
                        # Set new env var
                        os.environ[key] = value
                        config[key] = value

    return config


class ConfigLoader:
    """
    Advanced configuration loader with schema support
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

        # Load .env file first if available and requested
        if self.load_dotenv_first and DOTENV_AVAILABLE and self.dotenv_path:
            env_file = Path(self.dotenv_path)

            # Try multiple locations for .env file
            env_locations = []

            # 1. Exact path specified
            if env_file.is_absolute():
                env_locations.append(env_file)
            else:
                # 2. Current working directory
                env_locations.append(Path.cwd() / self.dotenv_path)

            # Load from the first .env file found
            for env_path in env_locations:
                if env_path.exists():
                    load_dotenv(env_path)
                    break

    def load_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable interpolation"""
        if not Path(yaml_path).exists():
            return {}

        # Load .env file if needed and not already loaded in __init__
        if self.load_dotenv_first and DOTENV_AVAILABLE and self.dotenv_path:
            env_file = Path(self.dotenv_path)

            # Try multiple locations for .env file
            env_locations = []

            # 1. Exact path specified
            if env_file.is_absolute():
                env_locations.append(env_file)
            else:
                # 2. Current working directory
                env_locations.append(Path.cwd() / self.dotenv_path)

                # 3. Same directory as YAML file
                yaml_dir = Path(yaml_path).parent
                env_locations.append(yaml_dir / self.dotenv_path)

            # Load from the first .env file found
            for env_path in env_locations:
                if env_path.exists():
                    load_dotenv(env_path)
                    break

        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

        if yaml_data:
            # Interpolate environment variables in the YAML data
            yaml_data = interpolate_env_vars(yaml_data)

        return yaml_data or {}

    def load_view(self, yaml_paths: Iterable[Union[str, Path]]) -> Dict[str, Any]:
        """
        Load and deep-merge multiple YAML files into a single configuration view.

        :param yaml_paths: Iterable of YAML file paths.
        :type yaml_paths: Iterable[str | pathlib.Path]
        :return: Composed configuration mapping.
        :rtype: dict[str, Any]
        """
        composed: Dict[str, Any] = {}
        for path in yaml_paths:
            data = self.load_from_yaml(path)
            if not isinstance(data, dict):
                raise ValueError("YAML configuration must be a mapping/object")
            composed = deep_merge_dicts(composed, data)
        return composed

    def load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        # Get all environment variables
        env_vars = dict(os.environ)

        # Convert back to nested structure
        return unflatten_env_vars(env_vars, self.prefix)

    def set_env_vars(self, config: Dict[str, Any], override: bool = False) -> None:
        """Set environment variables from configuration dictionary"""
        # Flatten the configuration
        flat_config = flatten_dict(config, self.prefix)

        # Set environment variables
        for key, value in flat_config.items():
            if override or key not in os.environ:
                os.environ[key] = value
