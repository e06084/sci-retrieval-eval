"""Experiment planning and run schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts.types import (
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
)
from eval_platform.benchmark import BenchmarkDatasetSpec, BenchmarkSettingSpec


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class ExperimentCorpusAssetConfig(BaseModel):
    """Dataset selection resolved from reusable corpus/index assets."""

    dataset_selection: str = "all"
    corpus_run_id: str
    bucket: str
    raw_prefix: str = "sciverse_benchmark/raw"
    s3_prefix: str = "sciverse_benchmark/assets"
    reuse_existing: bool = True
    expected_asset_fingerprints_by_slug: dict[str, dict[str, str]] = Field(
        default_factory=dict
    )
    required_artifact_types: list[str] = Field(
        default_factory=lambda: [
            NORMALIZED_DATASET_ARTIFACT_TYPE,
            ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
            MILVUS_COLLECTION_ARTIFACT_TYPE,
        ]
    )

    @field_validator(
        "dataset_selection",
        "corpus_run_id",
        "bucket",
        "raw_prefix",
        "s3_prefix",
    )
    @classmethod
    def validate_non_empty_strings(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("required_artifact_types")
    @classmethod
    def validate_required_artifact_types(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("required_artifact_types must not be empty")
        return list(dict.fromkeys(value))


class ExperimentRunConfig(BaseModel):
    """User-facing experiment configuration.

    An experiment expands to dataset x setting benchmark items, but the parent
    artifact is an experiment_run rather than a benchmark_suite_run.
    """

    experiment_run_id: str
    datasets: list[BenchmarkDatasetSpec] = Field(default_factory=list)
    corpus_assets: ExperimentCorpusAssetConfig | None = None
    settings: list[BenchmarkSettingSpec]
    metrics_k_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10, 20, 100, 1000]
    )
    query_limit: int | None = Field(default=None, gt=0)
    queries_per_shard: int = Field(default=1000, gt=0)
    reuse_existing: bool = True
    catalog_id: str | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_run_id")
    @classmethod
    def validate_experiment_run_id(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("catalog_id")
    @classmethod
    def validate_catalog_id(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("metrics_k_values")
    @classmethod
    def validate_metrics_k_values(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("metrics_k_values must not be empty")
        if any(k <= 0 for k in value):
            raise ValueError("metrics_k_values must contain only positive integers")
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_non_empty(self) -> ExperimentRunConfig:
        has_explicit_datasets = bool(self.datasets)
        has_corpus_assets = self.corpus_assets is not None
        if has_explicit_datasets == has_corpus_assets:
            raise ValueError("exactly one of datasets or corpus_assets must be provided")
        if not self.settings:
            raise ValueError("settings must not be empty")
        return self


class ExperimentStagePlan(BaseModel):
    """Planned action for one materialized child artifact."""

    stage: Literal["retrieval_run", "metrics_run", "benchmark_run"]
    action: Literal["reuse", "create"]
    artifact_type: str
    artifact_id: str
    generated_artifact_id: str
    asset_fingerprint_sha256: str | None = None
    reuse_reason: str | None = None


class ExperimentItemPlan(BaseModel):
    """Plan for one dataset x setting benchmark item."""

    dataset_key: str
    setting_key: str
    retrieval: ExperimentStagePlan
    metrics: ExperimentStagePlan
    benchmark: ExperimentStagePlan


class ExperimentPlan(BaseModel):
    """Dry-run plan for an experiment."""

    experiment_run_id: str
    item_count: int
    dataset_count: int
    setting_count: int
    reuse_existing: bool
    corpus_asset_plan: dict[str, Any] | None = None
    items: list[ExperimentItemPlan]


class ExperimentRunItemSummary(BaseModel):
    """Executed or reused item summary in an experiment_run."""

    dataset_key: str
    setting_key: str
    benchmark_run_artifact_id: str
    retrieval_run_artifact_id: str
    metrics_run_artifact_id: str
    actions: dict[str, str]
    main_score: float
    main_score_metric: str
    aggregate_metrics: dict[str, float]


class ExperimentRunSummary(BaseModel):
    """Summary file stored in an experiment_run artifact."""

    experiment_run_id: str
    item_count: int = Field(ge=0)
    dataset_count: int = Field(ge=0)
    setting_count: int = Field(ge=0)
    items: list[ExperimentRunItemSummary]

    @model_validator(mode="after")
    def validate_item_count(self) -> ExperimentRunSummary:
        if self.item_count != len(self.items):
            raise ValueError("item_count must match items length")
        return self
