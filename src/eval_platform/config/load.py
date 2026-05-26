"""Platform config loading and deep-merge helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from eval_platform.config.schema import PlatformConfig


class ConfigLoadError(Exception):
    """Raised when a platform config file or override is invalid."""


def deep_merge_config(base: Any, override: Any) -> Any:
    """Deep merge two config-like structures.

    Rules:
    - dict: recursive merge
    - list: replace
    - scalar: override
    - None: explicit override to None
    """
    if isinstance(base, dict) and isinstance(override, Mapping):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigLoadError("config file must contain a YAML mapping")
    return raw


def load_platform_config(
    config_path: Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> PlatformConfig:
    """Load platform config with precedence default < YAML < CLI overrides."""
    merged: dict[str, Any] = PlatformConfig().model_dump(mode="python")

    if config_path is not None:
        merged = deep_merge_config(merged, _load_yaml_mapping(config_path))

    if cli_overrides:
        merged = deep_merge_config(merged, dict(cli_overrides))

    return PlatformConfig.model_validate(merged)
