"""Tests for benchmark setting registry."""

from __future__ import annotations

import pytest

from eval_platform.benchmark import (
    DEFAULT_E1_E4_SETTINGS,
    BenchmarkSettingSpec,
    settings_for_selection,
)


def test_default_e1_e4_settings_order_and_values() -> None:
    assert [setting.setting_key for setting in DEFAULT_E1_E4_SETTINGS] == [
        "E1-milvus",
        "E2-es",
        "E3-hybrid",
        "E4-hybrid-rerank",
    ]
    assert [
        (
            setting.retrieval_mode,
            setting.sub_queries,
            setting.rewrite_enabled,
            setting.rerank_enabled,
        )
        for setting in DEFAULT_E1_E4_SETTINGS
    ] == [
        ("milvus", 0, False, False),
        ("es", 0, False, False),
        ("hybrid", 0, False, False),
        ("hybrid", 0, False, True),
    ]
    assert [setting.top_k for setting in DEFAULT_E1_E4_SETTINGS] == [100, 100, 100, 100]
    assert [setting.rerank_candidate_cap for setting in DEFAULT_E1_E4_SETTINGS] == [
        0,
        0,
        0,
        0,
    ]
    assert [setting.paper_cap for setting in DEFAULT_E1_E4_SETTINGS] == [0, 0, 0, 0]


def test_benchmark_setting_spec_uses_sciverse_v1_defaults() -> None:
    setting = BenchmarkSettingSpec(setting_key="demo", retrieval_mode="hybrid")

    assert setting.top_k == 100
    assert setting.rerank_candidate_cap == 0
    assert setting.paper_cap == 0


def test_settings_for_selection_supports_all_and_keys() -> None:
    assert settings_for_selection() == DEFAULT_E1_E4_SETTINGS
    assert settings_for_selection("all") == DEFAULT_E1_E4_SETTINGS
    assert [setting.setting_key for setting in settings_for_selection("E2-es")] == [
        "E2-es"
    ]
    assert [
        setting.setting_key
        for setting in settings_for_selection(["E3-hybrid", "E1-milvus"])
    ] == ["E3-hybrid", "E1-milvus"]


def test_settings_for_selection_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark setting key"):
        settings_for_selection("E5-rewrite")


def test_settings_for_selection_returns_copies_without_polluting_registry() -> None:
    settings_for_selection()[0].rerank_enabled = True
    assert settings_for_selection()[0].rerank_enabled is False
    assert settings_for_selection("all")[0].rerank_enabled is False

    settings_for_selection("E2-es")[0].rerank_enabled = True
    assert settings_for_selection("E2-es")[0].rerank_enabled is False

    selected = settings_for_selection(["E3-hybrid", "E1-milvus"])
    selected[0].rerank_enabled = True

    assert settings_for_selection()[2].rerank_enabled is False
    assert settings_for_selection(["E3-hybrid", "E1-milvus"])[0].rerank_enabled is False
