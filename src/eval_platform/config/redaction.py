"""Redaction helpers for platform configuration."""

from __future__ import annotations

from typing import Any

from eval_platform.config.schema import PlatformConfig

_SENSITIVE_TOKENS = ("password", "secret", "access_key", "api_key", "token")


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_TOKENS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def dump_redacted_config(config: PlatformConfig) -> dict[str, Any]:
    """Return a redacted config dump safe for logs, manifests, and reports."""
    return _redact_value(config.model_dump(mode="json"))
