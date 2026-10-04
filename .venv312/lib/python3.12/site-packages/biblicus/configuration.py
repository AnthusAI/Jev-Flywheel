"""
Configuration loading utilities for Biblicus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


def _parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_override_value(raw: str) -> object:
    """
    Parse a command-line override string into a Python value.

    :param raw: Raw override string.
    :type raw: str
    :return: Parsed value.
    :rtype: object
    """
    raw = str(raw)
    stripped = raw.strip()
    if not stripped:
        return ""
    if stripped[0] in {"{", "["}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return _parse_scalar(stripped)


def parse_dotted_overrides(pairs: Optional[List[str]]) -> Dict[str, object]:
    """
    Parse repeated key=value pairs into a dotted override mapping.

    :param pairs: Repeated command-line pairs.
    :type pairs: list[str] or None
    :return: Override mapping.
    :rtype: dict[str, object]
    :raises ValueError: If a pair is not key=value.
    """
    overrides: Dict[str, object] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Config values must be key=value (got {item!r})")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Config keys must be non-empty")
        overrides[key] = parse_override_value(raw)
    return overrides


def _set_dotted_key(target: MutableMapping[str, object], dotted_key: str, value: object) -> None:
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Override keys must be non-empty")
    current: MutableMapping[str, object] = target
    for part in parts[:-1]:
        existing = current.get(part)
        if not isinstance(existing, dict):
            nested: Dict[str, object] = {}
            current[part] = nested
            current = nested
        else:
            current = existing
    current[parts[-1]] = value


def apply_dotted_overrides(
    config: Dict[str, object], overrides: Mapping[str, object]
) -> Dict[str, object]:
    """
    Apply dotted key overrides to a nested configuration mapping.

    :param config: Base configuration mapping.
    :type config: dict[str, object]
    :param overrides: Dotted key override mapping.
    :type overrides: Mapping[str, object]
    :return: New configuration mapping with overrides applied.
    :rtype: dict[str, object]
    """
    updated: Dict[str, object] = json.loads(json.dumps(config))
    for key, value in overrides.items():
        _set_dotted_key(updated, key, value)
    return updated


def load_configuration_view(
    configuration_paths: Iterable[str],
    *,
    configuration_label: str = "Configuration",
    mapping_error_message: Optional[str] = None,
) -> Dict[str, object]:
    """
    Load a composed configuration view from one or more YAML files.

    :param configuration_paths: Iterable of configuration file paths in precedence order.
    :type configuration_paths: Iterable[str]
    :param configuration_label: Label used in error messages (for example: "Configuration file").
    :type configuration_label: str
    :return: Composed configuration view.
    :rtype: dict[str, object]
    :raises FileNotFoundError: If any configuration file is missing.
    :raises ValueError: If any configuration file is not a mapping/object.
    """
    from biblicus._vendor.dotyaml import load_yaml_view

    paths: List[str] = [str(path) for path in configuration_paths]
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_file():
            raise FileNotFoundError(f"{configuration_label} not found: {candidate}")
    try:
        view = load_yaml_view(paths)
    except ValueError as exc:
        message = mapping_error_message or f"{configuration_label} must be a mapping/object"
        raise ValueError(message) from exc
    return view
