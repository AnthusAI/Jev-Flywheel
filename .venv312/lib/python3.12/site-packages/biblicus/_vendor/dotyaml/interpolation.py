"""
Environment variable interpolation functionality for dotyaml.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Union


def interpolate_env_vars(data: Union[str, Dict[str, Any], Any]) -> Union[str, Dict[str, Any], Any]:
    """
    Recursively interpolate environment variables in YAML data using Jinja-like syntax.

    Supports syntax like: ``{{ ENV_VAR_NAME }}`` or ``{{ ENV_VAR_NAME|default_value }}``

    :param data: Data structure to interpolate (string, dict, list, etc).
    :type data: str or dict[str, Any] or Any
    :return: Data structure with environment variables interpolated.
    :rtype: str or dict[str, Any] or Any
    """

    if isinstance(data, str):
        return _interpolate_string(data)
    if isinstance(data, dict):
        return {key: interpolate_env_vars(value) for key, value in data.items()}
    if isinstance(data, list):
        return [interpolate_env_vars(item) for item in data]
    return data


def _interpolate_string(text: str) -> str:
    """
    Interpolate environment variables in a string using Jinja-like syntax.

    Supports:
    - ``{{ ENV_VAR }}`` required environment variable.
    - ``{{ ENV_VAR|default_value }}`` environment variable with default.

    :param text: String to interpolate.
    :type text: str
    :return: String with environment variables interpolated.
    :rtype: str
    :raises ValueError: If a required environment variable is not found.
    """

    pattern = r"\{\{\s*([A-Z_][A-Z0-9_]*)\s*(?:\|\s*([^}]*?))?\s*\}\}"

    def replace_match(match):  # type: ignore[no-untyped-def]
        env_var = match.group(1)
        default_value = match.group(2)

        env_value = os.getenv(env_var)

        if env_value is not None:
            return env_value
        if default_value is not None:
            return default_value.strip()
        raise ValueError(f"Required environment variable '{env_var}' not found")

    return re.sub(pattern, replace_match, text)
