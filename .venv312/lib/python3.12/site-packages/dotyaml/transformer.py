"""
YAML to environment variable transformation utilities
"""

import json
from typing import Any, Dict, List, Union


def flatten_dict(
    data: Dict[str, Any], prefix: str = "", separator: str = "_"
) -> Dict[str, str]:
    """
    Flatten a nested dictionary into dot-notation keys.

    Args:
        data: Nested dictionary to flatten
        prefix: Prefix to add to all keys
        separator: Separator to use between key parts

    Returns:
        Flattened dictionary with string values
    """
    result = {}

    for key, value in data.items():
        # Build the full key path
        if prefix:
            full_key = f"{prefix}{separator}{key.upper()}"
        else:
            full_key = key.upper()

        # Clean key: replace hyphens and dots with underscores
        clean_key = full_key.replace("-", "_").replace(".", "_")

        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            result.update(flatten_dict(value, clean_key, separator))
        else:
            # Convert value to string
            result[clean_key] = convert_value_to_string(value)

    return result


def convert_value_to_string(value: Any) -> str:
    """
    Convert a Python value to its environment variable string representation.

    Args:
        value: Value to convert

    Returns:
        String representation suitable for environment variables
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, (list, tuple)):
        # Convert list items to strings and join with commas
        string_items = [convert_value_to_string(item) for item in value]
        return ",".join(string_items)
    elif isinstance(value, dict):
        # For complex objects that couldn't be flattened, convert to JSON
        return json.dumps(value)
    else:
        # Fallback: convert to string
        return str(value)


def unflatten_env_vars(env_vars: Dict[str, str], prefix: str = "") -> Dict[str, Any]:
    """
    Convert flat environment variables back to nested dictionary structure.

    Args:
        env_vars: Dictionary of environment variables
        prefix: Prefix to filter by (if provided)

    Returns:
        Nested dictionary structure
    """
    result: Dict[str, Any] = {}

    for key, value in env_vars.items():
        # Filter by prefix if provided
        if prefix and not key.startswith(f"{prefix}_"):
            continue

        # Remove prefix
        clean_key = key
        if prefix:
            clean_key = key[len(prefix) + 1 :]  # +1 for underscore

        # Split key into parts
        parts = clean_key.lower().split("_")

        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value with type conversion
        final_key = parts[-1]
        current[final_key] = convert_string_to_value(value)

    return result


def convert_string_to_value(value: str) -> Any:
    """
    Convert string environment variable back to appropriate Python type.

    Args:
        value: String value from environment variable

    Returns:
        Converted value with appropriate type
    """
    if value == "":
        return None
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.isdigit():
        return int(value)
    elif value.replace(".", "").replace("-", "").isdigit():
        try:
            return float(value)
        except ValueError:
            return value
    elif "," in value:
        # Try to parse as comma-separated list
        items = [item.strip() for item in value.split(",")]
        return [convert_string_to_value(item) for item in items]
    else:
        # Try to parse as JSON, fallback to string
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
