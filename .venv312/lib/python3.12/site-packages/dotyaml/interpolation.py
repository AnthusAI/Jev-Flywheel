"""
Environment variable interpolation functionality for dotyaml
"""

import os
import re
from typing import Any, Dict, Union


def interpolate_env_vars(
    data: Union[str, Dict[str, Any], Any],
) -> Union[str, Dict[str, Any], Any]:
    """
    Recursively interpolate environment variables in YAML data using Jinja-like syntax.

    Supports syntax like: {{ ENV_VAR_NAME }} or {{ ENV_VAR_NAME|default_value }}

    Args:
        data: The data structure to interpolate (can be string, dict, list, etc.)

    Returns:
        The data structure with environment variables interpolated
    """
    if isinstance(data, str):
        return _interpolate_string(data)
    elif isinstance(data, dict):
        return {key: interpolate_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [interpolate_env_vars(item) for item in data]
    else:
        return data


def _interpolate_string(text: str) -> str:
    """
    Interpolate environment variables in a string using Jinja-like syntax.

    Supports:
    - {{ ENV_VAR }} - Required environment variable
    - {{ ENV_VAR|default_value }} - Environment variable with default

    Args:
        text: String to interpolate

    Returns:
        String with environment variables interpolated

    Raises:
        ValueError: If a required environment variable is not found
    """
    # Pattern to match {{ VAR_NAME }} or {{ VAR_NAME|default }}
    pattern = r"\{\{\s*([A-Z_][A-Z0-9_]*)\s*(?:\|\s*([^}]*?))?\s*\}\}"

    def replace_match(match: re.Match[str]) -> str:
        env_var = match.group(1)
        default_value = match.group(2)

        # Get environment variable value
        env_value = os.getenv(env_var)

        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value.strip()
        else:
            raise ValueError(f"Required environment variable '{env_var}' not found")

    return re.sub(pattern, replace_match, text)
