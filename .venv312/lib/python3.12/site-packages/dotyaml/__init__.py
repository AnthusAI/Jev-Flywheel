"""
dotyaml: Bridge YAML configuration files and environment variables

A simple library that allows applications to be configured via either YAML files
or environment variables, providing maximum deployment flexibility.
"""

__version__ = "1.1.1"

# Main API - keep it simple like python-dotenv
from .loader import ConfigLoader, load_config, load_yaml, load_yaml_view
from .interpolation import interpolate_env_vars

__all__ = [
    "ConfigLoader",
    "interpolate_env_vars",
    "load_config",
    "load_yaml",
    "load_yaml_view",
]
