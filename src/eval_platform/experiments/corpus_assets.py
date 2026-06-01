"""Resolve benchmark datasets from reusable corpus assets."""

from __future__ import annotations

from typing import Any

from eval_platform.artifacts import ArtifactStore
from eval_platform.artifacts.types import (
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
)
from eval_platform.benchmark import BenchmarkDatasetSpec
from eval_platform.corpus_assets import (
    build_plan_for_datasets,
    dataset_specs_for_selection,
    inventory_corpus_assets,
)
from eval_platform.experiments.schema import ExperimentCorpusAssetConfig


class ExperimentCorpusAssetResolutionError(Exception):
    """Raised when corpus asset selection cannot produce benchmark datasets."""


def resolve_benchmark_datasets_from_corpus_assets(
    store: ArtifactStore,
    config: ExperimentCorpusAssetConfig,
) -> tuple[list[BenchmarkDatasetSpec], dict[str, Any]]:
    """Resolve dataset slugs to benchmark-ready artifact ids and resource names.

    The experiment layer only consumes corpus assets. Missing corpus asset stages
    are surfaced in the returned plan/error; corpus construction remains owned by
    the corpus asset preparation flow.
    """

    dataset_specs = dataset_specs_for_selection(config.dataset_selection)
    inventory = inventory_corpus_assets(
        store=store,
        raw_client=_AlwaysExistsRawClient(),
        bucket=config.bucket,
        raw_prefix=config.raw_prefix,
        datasets=dataset_specs,
    )
    plan = build_plan_for_datasets(
        datasets=dataset_specs,
        run_id=config.corpus_run_id,
        bucket=config.bucket,
        raw_prefix=config.raw_prefix,
        s3_prefix=config.s3_prefix,
        raw_exists_by_slug={spec.slug: True for spec in dataset_specs},
        reuse_existing=config.reuse_existing,
        inventory=inventory,
        expected_asset_fingerprints_by_slug=config.expected_asset_fingerprints_by_slug,
    )
    return (
        benchmark_dataset_specs_from_corpus_asset_plan(
            plan,
            required_artifact_types=config.required_artifact_types,
        ),
        plan,
    )


def benchmark_dataset_specs_from_corpus_asset_plan(
    plan: dict[str, Any],
    *,
    required_artifact_types: list[str] | None = None,
) -> list[BenchmarkDatasetSpec]:
    """Convert a corpus asset plan into benchmark dataset specs."""

    required = required_artifact_types or [
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        MILVUS_COLLECTION_ARTIFACT_TYPE,
    ]
    missing_by_dataset: dict[str, list[str]] = {}
    datasets: list[BenchmarkDatasetSpec] = []

    for task_name, dataset_plan in plan.get("datasets", {}).items():
        steps_by_type = {
            step["artifact_type"]: step
            for step in dataset_plan.get("steps", [])
            if "artifact_type" in step
        }
        missing = [
            artifact_type
            for artifact_type in required
            if steps_by_type.get(artifact_type, {}).get("action") != "reuse"
        ]
        if missing:
            missing_by_dataset[str(task_name)] = missing
            continue

        resolved_artifact_ids = dataset_plan.get("resolved_artifact_ids", {})
        resolved_resource_names = dataset_plan.get("resolved_resource_names", {})
        datasets.append(
            BenchmarkDatasetSpec(
                dataset_key=str(dataset_plan["slug"]),
                task_name=str(task_name),
                normalized_dataset_artifact_id=_required_plan_value(
                    resolved_artifact_ids,
                    NORMALIZED_DATASET_ARTIFACT_TYPE,
                    task_name,
                ),
                elasticsearch_index_artifact_id=_required_plan_value(
                    resolved_artifact_ids,
                    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
                    task_name,
                ),
                milvus_collection_artifact_id=_required_plan_value(
                    resolved_artifact_ids,
                    MILVUS_COLLECTION_ARTIFACT_TYPE,
                    task_name,
                ),
                index_name=_required_plan_value(
                    resolved_resource_names,
                    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
                    task_name,
                ),
                collection_name=_required_plan_value(
                    resolved_resource_names,
                    MILVUS_COLLECTION_ARTIFACT_TYPE,
                    task_name,
                ),
                metadata={
                    "corpus_asset_plan_run_id": plan.get("run_id"),
                    "corpus_asset_slug": dataset_plan.get("slug"),
                    "resolved_artifact_ids": resolved_artifact_ids,
                    "resolved_resource_names": resolved_resource_names,
                },
            )
        )

    if missing_by_dataset:
        raise ExperimentCorpusAssetResolutionError(
            "Corpus assets are not complete for experiment use: "
            f"{missing_by_dataset}"
        )
    return datasets


def _required_plan_value(
    values: dict[str, Any],
    key: str,
    task_name: object,
) -> str:
    value = values.get(key)
    if isinstance(value, str) and value.strip():
        return value
    raise ExperimentCorpusAssetResolutionError(
        f"Corpus asset plan for {task_name!r} is missing {key!r}"
    )


class _AlwaysExistsRawClient:
    """Small stand-in because experiment planning only needs corpus manifests."""

    def list_objects_v2(self, **_: Any) -> dict[str, Any]:
        return {"Contents": [{"Key": "__experiment_raw_prefix_placeholder__"}]}
