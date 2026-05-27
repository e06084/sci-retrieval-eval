"""Tests for inventory_real_corpus_assets script wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import inventory_real_corpus_assets as inventory_script  # noqa: E402


def test_inventory_script_run_delegates_to_corpus_asset_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(s3=SimpleNamespace(bucket="bucket"))
    client = object()
    store = object()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        inventory_script,
        "load_config_and_client",
        lambda path: (config, client),
    )
    monkeypatch.setattr(
        inventory_script,
        "make_s3_artifact_store",
        lambda *, config, s3_prefix, client: store,
    )
    monkeypatch.setattr(
        inventory_script,
        "inventory_corpus_assets",
        lambda *, store, raw_client, bucket, raw_prefix: {
            "datasets": {},
            "artifact_stage_order": [],
        },
    )
    monkeypatch.setattr(
        inventory_script,
        "redacted_config_summary",
        lambda config: {"safe": True},
    )
    monkeypatch.setattr(
        inventory_script,
        "output_payload",
        lambda payload, output: captured.update(payload=payload, output=output),
    )

    payload = inventory_script.run(
        argparse.Namespace(
            config=Path("config.yaml"),
            s3_prefix="prefix",
            raw_prefix="raw",
            output=None,
        )
    )

    assert payload["kind"] == "five_dataset_corpus_asset_inventory"
    assert payload["s3_bucket"] == "bucket"
    assert payload["artifact_prefix"] == "prefix"
    assert payload["raw_prefix"] == "raw"
    assert payload["config"] == {"safe": True}
    assert captured["payload"] == payload
