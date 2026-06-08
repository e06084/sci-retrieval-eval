"""Benchmark setting registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.defaults import (
    DEFAULT_HYBRID_PER_SOURCE_TOPK,
    DEFAULT_PAPER_CAP,
    DEFAULT_RERANK_CANDIDATE_CAP,
    DEFAULT_RERANK_CROSS_PATH_TOPK,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RRF_PATH_TOPK,
)


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class BenchmarkSettingSpec(BaseModel):
    """Retrieval setting used by a benchmark suite item."""

    setting_key: str
    retrieval_mode: Literal["es", "milvus", "hybrid"]
    sub_queries: int = Field(default=0, ge=0)
    rewrite_enabled: bool = False
    rerank_enabled: bool = False
    top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, gt=0)
    hybrid_per_source_topk: int = Field(default=DEFAULT_HYBRID_PER_SOURCE_TOPK, gt=0)
    rrf_path_topk: int = Field(default=DEFAULT_RRF_PATH_TOPK, gt=0)
    paper_cap: int = Field(default=DEFAULT_PAPER_CAP, ge=0)
    rerank_cross_path_topk: int = Field(default=DEFAULT_RERANK_CROSS_PATH_TOPK, ge=0)
    rerank_candidate_cap: int = Field(default=DEFAULT_RERANK_CANDIDATE_CAP, ge=0)
    trace_mode: Literal["replay", "light", "none"] = "replay"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("setting_key")
    @classmethod
    def validate_setting_key(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


DEFAULT_E1_E4_SETTINGS = [
    BenchmarkSettingSpec(
        setting_key="E1-milvus",
        retrieval_mode="milvus",
        sub_queries=0,
        rewrite_enabled=False,
        rerank_enabled=False,
    ),
    BenchmarkSettingSpec(
        setting_key="E2-es",
        retrieval_mode="es",
        sub_queries=0,
        rewrite_enabled=False,
        rerank_enabled=False,
    ),
    BenchmarkSettingSpec(
        setting_key="E3-hybrid",
        retrieval_mode="hybrid",
        sub_queries=0,
        rewrite_enabled=False,
        rerank_enabled=False,
    ),
    BenchmarkSettingSpec(
        setting_key="E4-hybrid-rerank",
        retrieval_mode="hybrid",
        sub_queries=0,
        rewrite_enabled=False,
        rerank_enabled=True,
    ),
]


def settings_for_selection(
    selection: Sequence[str] | str | None = None,
) -> list[BenchmarkSettingSpec]:
    """Return default E1-E4 settings, optionally filtered by setting key."""

    if selection is None or selection == "all":
        return [setting.model_copy(deep=True) for setting in DEFAULT_E1_E4_SETTINGS]
    selected_keys = [selection] if isinstance(selection, str) else list(selection)
    by_key = {setting.setting_key: setting for setting in DEFAULT_E1_E4_SETTINGS}
    out: list[BenchmarkSettingSpec] = []
    for key in selected_keys:
        if key not in by_key:
            raise ValueError(f"Unknown benchmark setting key: {key}")
        out.append(by_key[key].model_copy(deep=True))
    return out
