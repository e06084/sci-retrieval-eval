"""Tests for platform config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.config import PlatformConfig, deep_merge_config, load_platform_config


def test_platform_config_constructs() -> None:
    config = PlatformConfig()

    assert config.s3 is not None
    assert config.embedding is not None
    assert config.raw_sources == {}


def test_yaml_load_applies_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "s3:\n"
        "  bucket: example-bucket\n"
        "embedding:\n"
        "  model: test-model\n"
        "  batch_size: 8\n",
        encoding="utf-8",
    )

    loaded = load_platform_config(config_path)

    assert loaded.s3.bucket == "example-bucket"
    assert loaded.embedding.model == "test-model"
    assert loaded.embedding.batch_size == 8


def test_cli_overrides_take_precedence_over_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "embedding:\n"
        "  batch_size: 8\n"
        "s3:\n"
        "  prefix: from-yaml\n",
        encoding="utf-8",
    )

    loaded = load_platform_config(
        config_path,
        cli_overrides={"embedding": {"batch_size": 16}, "s3": {"prefix": "from-cli"}},
    )

    assert loaded.embedding.batch_size == 16
    assert loaded.s3.prefix == "from-cli"


def test_environment_variables_do_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "from-env")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("embedding:\n  model: from-yaml\n", encoding="utf-8")

    loaded = load_platform_config(config_path)

    assert loaded.embedding.model == "from-yaml"


def test_deep_merge_nested_dict_and_list_replace() -> None:
    merged = deep_merge_config(
        {
            "embedding": {
                "batch_size": 8,
                "endpoints": [{"url": "a", "api_key": "x"}],
            },
            "search_runtime": {"rewrite": {"enabled": False, "model": "m1"}},
        },
        {
            "embedding": {
                "endpoints": [{"url": "b", "api_key": "y"}],
            },
            "search_runtime": {"rewrite": {"enabled": True}},
        },
    )

    assert merged["embedding"]["batch_size"] == 8
    assert merged["embedding"]["endpoints"] == [{"url": "b", "api_key": "y"}]
    assert merged["search_runtime"]["rewrite"] == {"enabled": True, "model": "m1"}


def test_none_explicitly_overrides_value() -> None:
    merged = deep_merge_config({"s3": {"prefix": "abc"}}, {"s3": {"prefix": None}})

    assert merged["s3"]["prefix"] is None


def test_unknown_yaml_field_raises_validation_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "embedding:\n"
        "  batch_szie: 8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_platform_config(config_path)


def test_unknown_cli_override_field_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        load_platform_config(cli_overrides={"embedding": {"batch_szie": 8}})
