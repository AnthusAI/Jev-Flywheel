"""
YAML to environment variable transformation utilities for dotyaml.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def flatten_dict(data: Dict[str, Any], prefix: str = "", separator: str = "_") -> Dict[str, str]:
    """
    Flatten a nested dictionary into environment-variable style keys.

    :param data: Nested dictionary to flatten.
    :type data: dict[str, Any]
    :param prefix: Prefix to add to all keys.
    :type prefix: str
    :param separator: Separator between key parts.
    :type separator: str
    :return: Flattened mapping with string values.
    :rtype: dict[str, str]
    """

    result: Dict[str, str] = {}

    for key, value in data.items():
        if prefix:
            full_key = f"{prefix}{separator}{key.upper()}"
        else:
            full_key = key.upper()

        clean_key = full_key.replace("-", "_").replace(".", "_")

        if isinstance(value, dict):
            result.update(flatten_dict(value, clean_key, separator))
        else:
            result[clean_key] = convert_value_to_string(value)

    return result


def convert_value_to_string(value: Any) -> str:
    """
    Convert a Python value to its environment variable string representation.

    :param value: Value to convert.
    :type value: Any
    :return: String representation suitable for environment variables.
    :rtype: str
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(convert_value_to_string(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def unflatten_env_vars(env_vars: Dict[str, str], prefix: str = "") -> Dict[str, Any]:
    """
    Convert flat environment variables back to nested dictionary structure.

    :param env_vars: Mapping of environment variables.
    :type env_vars: dict[str, str]
    :param prefix: Optional prefix to filter by.
    :type prefix: str
    :return: Nested dictionary structure.
    :rtype: dict[str, Any]
    """

    result: Dict[str, Any] = {}

    for key, value in env_vars.items():
        if prefix and not key.startswith(f"{prefix}_"):
            continue

        clean_key = key
        if prefix:
            clean_key = key[len(prefix) + 1 :]

        parts = clean_key.lower().split("_")

        current: Dict[str, Any] = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        current[final_key] = convert_string_to_value(value)

    return result


def convert_string_to_value(value: str) -> Any:
    """
    Convert a string environment variable back to an appropriate Python type.

    :param value: String value from an environment variable.
    :type value: str
    :return: Converted Python value.
    :rtype: Any
    """

    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    if value.replace(".", "").replace("-", "").isdigit():
        try:
            return float(value)
        except ValueError:
            return value
    if "," in value:
        items = [item.strip() for item in value.split(",")]
        return [convert_string_to_value(item) for item in items]
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
