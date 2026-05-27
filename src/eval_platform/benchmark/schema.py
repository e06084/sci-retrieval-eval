"""Benchmark run schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.metrics import MetricsRunConfig
from eval_platform.retrieval import RetrievalRunConfig


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class BenchmarkRunConfig(BaseModel):
    """Configuration for the minimal benchmark runner."""

    output_artifact_id: str
    source_normalized_dataset_artifact_id: str
    retrieval: RetrievalRunConfig
    metrics: MetricsRunConfig
    setting_name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("output_artifact_id", "source_normalized_dataset_artifact_id")
    @classmethod
    def validate_required_ids(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("setting_name", "description")
    @classmethod
    def validate_optional_strings(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for tag in value:
            normalized = tag.strip()
            if not normalized or normalized in seen:
                continue
            out.append(normalized)
            seen.add(normalized)
        return out

    @model_validator(mode="after")
    def validate_nested_configs(self) -> BenchmarkRunConfig:
        if (
            self.retrieval.source_normalized_dataset_artifact_id
            != self.source_normalized_dataset_artifact_id
        ):
            raise ValueError(
                "retrieval.source_normalized_dataset_artifact_id must match "
                "source_normalized_dataset_artifact_id"
            )
        if (
            self.metrics.source_normalized_dataset_artifact_id
            != self.source_normalized_dataset_artifact_id
        ):
            raise ValueError(
                "metrics.source_normalized_dataset_artifact_id must match "
                "source_normalized_dataset_artifact_id"
            )
        if self.metrics.source_retrieval_run_artifact_id != self.retrieval.output_artifact_id:
            raise ValueError(
                "metrics.source_retrieval_run_artifact_id must match retrieval.output_artifact_id"
            )
        return self


class BenchmarkRunSummary(BaseModel):
    """Summary file stored in a benchmark_run artifact."""

    benchmark_run_artifact_id: str
    setting_name: str | None = None
    retrieval_run_artifact_id: str
    metrics_run_artifact_id: str
    source_normalized_dataset_artifact_id: str
    main_score: float
    main_score_metric: str
    aggregate_metrics: dict[str, float]

    @field_validator(
        "benchmark_run_artifact_id",
        "retrieval_run_artifact_id",
        "metrics_run_artifact_id",
        "source_normalized_dataset_artifact_id",
        "main_score_metric",
    )
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")
