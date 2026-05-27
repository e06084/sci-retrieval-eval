"""Tests for corpus asset S3/output helpers."""

from __future__ import annotations

from eval_platform.corpus_assets import safe_json_dumps


def test_safe_json_dumps_redacts_secrets() -> None:
    payload = {
        "s3": {"access_key_id": "abc", "secret_access_key": "def"},
        "headers": {"Authorization": "Bearer token"},
        "nested": [{"api_key": "secret"}, {"ok": "value"}],
    }

    text = safe_json_dumps(payload)

    assert "abc" not in text
    assert "def" not in text
    assert "Bearer token" not in text
    assert '"***"' in text
