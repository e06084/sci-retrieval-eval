"""Benchmark suite configuration and runner."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.artifacts.types import BENCHMARK_RUN_ARTIFACT_TYPE
from eval_platform.benchmark.artifact import read_benchmark_run_artifact
from eval_platform.benchmark.runner import run_benchmark
from eval_platform.benchmark.schema import BenchmarkRunConfig
from eval_platform.benchmark.settings import BenchmarkSettingSpec
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.embeddings import EmbeddingClient
from eval_platform.metrics import MetricsRunConfig
from eval_platform.retrieval import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RetrievalRunConfig,
    RewriteClient,
)

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_key_component(value: str, field_name: str) -> str:
    value = _non_empty_string(value, field_name)
    if not _KEY_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


class BenchmarkDatasetSpec(BaseModel):
    """Dataset/index assets needed by one benchmark suite dataset."""

    dataset_key: str
    task_name: str | None = None
    normalized_dataset_artifact_id: str
    elasticsearch_index_artifact_id: str
    milvus_collection_artifact_id: str
    index_name: str
    collection_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_key")
    @classmethod
    def validate_dataset_key(cls, value: str, info: ValidationInfo) -> str:
        return _validate_key_component(value, info.field_name or "field")

    @field_validator(
        "normalized_dataset_artifact_id",
        "elasticsearch_index_artifact_id",
        "milvus_collection_artifact_id",
        "index_name",
        "collection_name",
    )
    @classmethod
    def validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")


class BenchmarkSuiteRunConfig(BaseModel):
    """Configuration for running a dataset x setting benchmark suite."""

    suite_run_id: str
    datasets: list[BenchmarkDatasetSpec]
    settings: list[BenchmarkSettingSpec]
    metrics_k_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10, 20, 100, 1000]
    )
    query_limit: int | None = Field(default=None, gt=0)
    queries_per_shard: int = Field(default=1000, gt=0)
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("suite_run_id")
    @classmethod
    def validate_suite_run_id(cls, value: str, info: ValidationInfo) -> str:
        return _validate_key_component(value, info.field_name or "field")

    @field_validator("metrics_k_values")
    @classmethod
    def validate_metrics_k_values(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("metrics_k_values must not be empty")
        if any(k <= 0 for k in value):
            raise ValueError("metrics_k_values must contain only positive integers")
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_non_empty_and_unique_keys(self) -> BenchmarkSuiteRunConfig:
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        if not self.settings:
            raise ValueError("settings must not be empty")
        _validate_unique_keys(
            [dataset.dataset_key for dataset in self.datasets],
            "dataset_key",
        )
        _validate_unique_keys(
            [setting.setting_key for setting in self.settings],
            "setting_key",
        )
        for setting in self.settings:
            _validate_key_component(setting.setting_key, "setting_key")
        return self


class BenchmarkSuiteItemSummary(BaseModel):
    """Summary for one dataset x setting benchmark suite item."""

    dataset_key: str
    setting_key: str
    benchmark_run_artifact_id: str
    retrieval_run_artifact_id: str
    metrics_run_artifact_id: str
    main_score: float
    main_score_metric: str
    aggregate_metrics: dict[str, float]

    @field_validator(
        "dataset_key",
        "setting_key",
        "benchmark_run_artifact_id",
        "retrieval_run_artifact_id",
        "metrics_run_artifact_id",
        "main_score_metric",
    )
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class BenchmarkSuiteRunSummary(BaseModel):
    """Summary file stored in a benchmark_suite_run artifact."""

    suite_run_id: str
    item_count: int = Field(ge=0)
    dataset_count: int = Field(ge=0)
    setting_count: int = Field(ge=0)
    items: list[BenchmarkSuiteItemSummary]

    @field_validator("suite_run_id")
    @classmethod
    def validate_summary_suite_run_id(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_item_count(self) -> BenchmarkSuiteRunSummary:
        if self.item_count != len(self.items):
            raise ValueError("item_count must match items length")
        return self


def build_benchmark_run_config(
    suite_config: BenchmarkSuiteRunConfig,
    dataset: BenchmarkDatasetSpec,
    setting: BenchmarkSettingSpec,
) -> BenchmarkRunConfig:
    """Build one child benchmark_run config for a dataset x setting pair."""

    base_id = f"{suite_config.suite_run_id}__{dataset.dataset_key}__{setting.setting_key}"
    retrieval_id = f"{base_id}__retrieval"
    metrics_id = f"{base_id}__metrics"
    benchmark_id = f"{base_id}__benchmark"
    uses_milvus = setting.retrieval_mode in {"milvus", "hybrid"}
    retrieval = RetrievalRunConfig(
        source_normalized_dataset_artifact_id=dataset.normalized_dataset_artifact_id,
        output_artifact_id=retrieval_id,
        retrieval_mode=setting.retrieval_mode,
        top_k=setting.top_k,
        query_limit=suite_config.query_limit,
        queries_per_shard=suite_config.queries_per_shard,
        trace_mode=setting.trace_mode,
        elasticsearch_index_artifact_id=dataset.elasticsearch_index_artifact_id,
        milvus_collection_artifact_id=(
            dataset.milvus_collection_artifact_id if uses_milvus else None
        ),
        index_name=dataset.index_name,
        collection_name=dataset.collection_name if uses_milvus else None,
        sub_queries=setting.sub_queries,
        rewrite_enabled=setting.rewrite_enabled,
        rerank_enabled=setting.rerank_enabled,
        hybrid_per_source_topk=setting.hybrid_per_source_topk,
        rrf_path_topk=setting.rrf_path_topk,
        rerank_cross_path_topk=setting.rerank_cross_path_topk,
        rerank_candidate_cap=setting.rerank_candidate_cap,
        created_by=suite_config.created_by,
        code_git_sha=suite_config.code_git_sha,
        metadata=_item_metadata(suite_config, dataset, setting),
    )
    metrics = MetricsRunConfig(
        source_normalized_dataset_artifact_id=dataset.normalized_dataset_artifact_id,
        source_retrieval_run_artifact_id=retrieval_id,
        output_artifact_id=metrics_id,
        k_values=suite_config.metrics_k_values,
        queries_per_shard=suite_config.queries_per_shard,
        created_by=suite_config.created_by,
        code_git_sha=suite_config.code_git_sha,
        metadata=_item_metadata(suite_config, dataset, setting),
    )
    return BenchmarkRunConfig(
        output_artifact_id=benchmark_id,
        source_normalized_dataset_artifact_id=dataset.normalized_dataset_artifact_id,
        retrieval=retrieval,
        metrics=metrics,
        setting_name=setting.setting_key,
        tags=[suite_config.suite_run_id, dataset.dataset_key, setting.setting_key],
        created_by=suite_config.created_by,
        code_git_sha=suite_config.code_git_sha,
        metadata=_item_metadata(suite_config, dataset, setting),
    )


def run_benchmark_suite(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: BenchmarkSuiteRunConfig,
    *,
    es_client: ElasticsearchRetrievalClient | None = None,
    milvus_client: MilvusRetrievalClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    rewrite_client: RewriteClient | None = None,
    rerank_client: RerankClient | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run all dataset x setting benchmark items and write a suite summary."""

    item_summaries: list[BenchmarkSuiteItemSummary] = []
    dependencies: list[ArtifactDependency] = []
    total_items = len(config.datasets) * len(config.settings)
    completed_items = 0
    report_progress(
        progress_reporter,
        stage="benchmark_suite_run",
        current=0,
        total=total_items,
        message="Starting benchmark suite",
        metadata=_suite_progress_metadata(config),
    )
    for dataset in config.datasets:
        for setting in config.settings:
            item_config = build_benchmark_run_config(config, dataset, setting)
            report_progress(
                progress_reporter,
                stage="benchmark_suite_run",
                current=completed_items,
                total=total_items,
                message="Starting benchmark suite item",
                metadata={
                    **_suite_progress_metadata(config),
                    **_item_progress_metadata(dataset, setting, item_config),
                    "item_index": completed_items + 1,
                },
            )
            run_benchmark(
                source_store,
                output_store,
                item_config,
                es_client=es_client,
                milvus_client=milvus_client,
                embedding_client=embedding_client,
                rewrite_client=rewrite_client,
                rerank_client=rerank_client,
                progress_reporter=progress_reporter,
            )
            child_summary = read_benchmark_run_artifact(
                output_store,
                item_config.output_artifact_id,
            )
            item_summaries.append(
                BenchmarkSuiteItemSummary(
                    dataset_key=dataset.dataset_key,
                    setting_key=setting.setting_key,
                    benchmark_run_artifact_id=item_config.output_artifact_id,
                    retrieval_run_artifact_id=child_summary.retrieval_run_artifact_id,
                    metrics_run_artifact_id=child_summary.metrics_run_artifact_id,
                    main_score=child_summary.main_score,
                    main_score_metric=child_summary.main_score_metric,
                    aggregate_metrics=child_summary.aggregate_metrics,
                )
            )
            dependencies.append(
                ArtifactDependency(
                    artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
                    artifact_id=item_config.output_artifact_id,
                )
            )
            completed_items += 1
            report_progress(
                progress_reporter,
                stage="benchmark_suite_run",
                current=completed_items,
                total=total_items,
                message="Completed benchmark suite item",
                metadata={
                    **_suite_progress_metadata(config),
                    **_item_progress_metadata(dataset, setting, item_config),
                    "item_index": completed_items,
                    "main_score": child_summary.main_score,
                    "main_score_metric": child_summary.main_score_metric,
                },
            )

    summary = BenchmarkSuiteRunSummary(
        suite_run_id=config.suite_run_id,
        item_count=len(item_summaries),
        dataset_count=len(config.datasets),
        setting_count=len(config.settings),
        items=item_summaries,
    )
    from eval_platform.benchmark.suite_artifact import write_benchmark_suite_run_artifact

    return write_benchmark_suite_run_artifact(
        output_store,
        config.suite_run_id,
        summary,
        metadata=_suite_manifest_metadata(config),
        dependencies=dependencies,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )


def _validate_unique_keys(values: list[str], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value}")
        seen.add(value)


def _item_metadata(
    suite_config: BenchmarkSuiteRunConfig,
    dataset: BenchmarkDatasetSpec,
    setting: BenchmarkSettingSpec,
) -> dict[str, Any]:
    return {
        "suite_run_id": suite_config.suite_run_id,
        "dataset_key": dataset.dataset_key,
        "setting_key": setting.setting_key,
        "task_name": dataset.task_name,
        "dataset_metadata": dataset.metadata,
        "setting_metadata": setting.metadata,
    }


def _suite_manifest_metadata(config: BenchmarkSuiteRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "suite_run_id": config.suite_run_id,
            "dataset_count": len(config.datasets),
            "setting_count": len(config.settings),
            "item_count": len(config.datasets) * len(config.settings),
            "query_limit": config.query_limit,
            "datasets": [
                dataset.model_dump(mode="json")
                for dataset in config.datasets
            ],
            "settings": [
                setting.model_dump(mode="json")
                for setting in config.settings
            ],
        }
    )
    return metadata


def _suite_progress_metadata(config: BenchmarkSuiteRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "suite_run_id": config.suite_run_id,
            "dataset_count": len(config.datasets),
            "setting_count": len(config.settings),
            "item_count": len(config.datasets) * len(config.settings),
            "query_limit": config.query_limit,
        }
    )
    return metadata


def _item_progress_metadata(
    dataset: BenchmarkDatasetSpec,
    setting: BenchmarkSettingSpec,
    item_config: BenchmarkRunConfig,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset.dataset_key,
        "setting_key": setting.setting_key,
        "benchmark_run_artifact_id": item_config.output_artifact_id,
        "retrieval_run_artifact_id": item_config.retrieval.output_artifact_id,
        "metrics_run_artifact_id": item_config.metrics.output_artifact_id,
    }
