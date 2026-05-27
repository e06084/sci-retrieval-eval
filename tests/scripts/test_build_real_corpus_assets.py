"""Tests for build_real_corpus_assets script wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eval_platform.corpus_assets import CorpusAssetError

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_real_corpus_assets as build_script  # noqa: E402


def test_build_script_run_delegates_to_corpus_asset_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(s3=SimpleNamespace(bucket="bucket"))
    client = object()
    store = object()
    spec = SimpleNamespace(slug="ifir_nfcorpus")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        build_script,
        "load_config_and_client",
        lambda path: (config, client),
    )
    monkeypatch.setattr(
        build_script,
        "dataset_specs_for_selection",
        lambda selection: [spec],
    )
    monkeypatch.setattr(
        build_script,
        "make_s3_artifact_store",
        lambda *, config, s3_prefix, client: store,
    )
    monkeypatch.setattr(
        build_script,
        "inventory_corpus_assets",
        lambda *, store, raw_client, bucket, raw_prefix, datasets: {
            "datasets": {"IFIRNFCorpus": {}}
        },
    )
    monkeypatch.setattr(build_script, "raw_prefix_key", lambda raw_prefix, spec: "raw/key")
    monkeypatch.setattr(build_script, "raw_prefix_exists", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        build_script,
        "build_plan_for_datasets",
        lambda **kwargs: {"mode": "dry_run", "datasets": {"IFIRNFCorpus": {}}},
    )
    monkeypatch.setattr(build_script, "redacted_config_summary", lambda config: {"safe": True})
    monkeypatch.setattr(
        build_script,
        "output_payload",
        lambda payload, output: captured.update(payload=payload, output=output),
    )

    payload = build_script.run(
        argparse.Namespace(
            config=Path("config.yaml"),
            dataset="IFIRNFCorpus",
            run_id="run_001",
            s3_prefix="prefix",
            raw_prefix="raw",
            reuse_existing=True,
            dry_run=True,
            execute=False,
            output=None,
        )
    )

    assert payload["kind"] == "five_dataset_corpus_asset_build_plan"
    assert payload["execute"] is False
    assert payload["config"] == {"safe": True}
    assert payload["inventory"] == {"datasets": {"IFIRNFCorpus": {}}}
    assert captured["payload"] == payload


def test_build_script_refuses_execute() -> None:
    with pytest.raises(CorpusAssetError, match="--execute is intentionally not implemented"):
        build_script.run(
            argparse.Namespace(
                execute=True,
                config=Path("config.yaml"),
                dataset="IFIRNFCorpus",
                run_id="run_001",
                s3_prefix="prefix",
                raw_prefix="raw",
                reuse_existing=False,
                dry_run=False,
                output=None,
            )
        )
