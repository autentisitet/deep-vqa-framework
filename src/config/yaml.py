"""Shared YAML mapping loading and nested overrides for training and serving."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_yaml_mapping(path: Path) -> dict:
    """Read a YAML mapping, rejecting malformed or non-mapping configuration."""
    with path.open(encoding="utf-8") as handle:
        try:
            settings = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return settings


def deep_merge(base: dict, override: dict) -> dict:
    """Merge nested mappings without sharing mutable values with either input.

    Scalars, lists and explicit nulls replace the previous value. Schema
    validation belongs to the caller and runs after all overrides are applied.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
