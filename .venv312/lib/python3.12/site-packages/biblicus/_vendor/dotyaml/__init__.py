"""
Vendored dotyaml utilities.

This package vendors the minimal pieces of the `dotyaml` project that Biblicus uses for
loading and interpolating YAML configuration files.
"""

from __future__ import annotations

from .interpolation import interpolate_env_vars
from .loader import ConfigLoader, load_config, load_yaml_view

__all__ = ["ConfigLoader", "interpolate_env_vars", "load_config", "load_yaml_view"]
