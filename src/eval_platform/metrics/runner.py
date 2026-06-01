"""Metrics run orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.assets import (
    add_asset_fingerprint_metadata,
    build_asset_fingerprint,
    manifest_asset_fingerprint_sha256,
    metrics_run_fingerprint_components,
)
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.datasets import (
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    NormalizedDataset,
    read_normalized_dataset_artifact,
)
from eval_platform.metrics.artifact import METRICS_RUN_ARTIFACT_TYPE, write_metrics_run_artifact
from eval_platform.metrics.ir import aggregate_query_metrics, compute_query_metrics
from eval_platform.metrics.projection import project_retrieval_result_to_docs
from eval_platform.metrics.schema import (
    MetricsRunData,
    ProjectionStats,
    QueryMetricsRecord,
    RankedDoc,
)
from eval_platform.retrieval import (
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    RetrievalQueryResult,
    read_retrieval_run_artifact,
)

MAIN_SCORE_METRIC = "ndcg_at_10"


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class MetricsRunConfig(BaseModel):
    """Configuration for computing a metrics_run artifact."""

    source_normalized_dataset_artifact_id: str
    source_retrieval_run_artifact_id: str
    output_artifact_id: str
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20, 100, 1000])
    doc_aggregation: Literal["first_chunk_rank"] = "first_chunk_rank"
    doc_score: Literal["reciprocal_first_chunk_rank"] = "reciprocal_first_chunk_rank"
    queries_per_shard: int = Field(default=1000, gt=0)
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_normalized_dataset_artifact_id",
        "source_retrieval_run_artifact_id",
        "output_artifact_id",
    )
    @classmethod
    def validate_required_ids(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("k_values must not be empty")
        if any(k <= 0 for k in value):
            raise ValueError("k_values must contain only positive integers")
        return sorted(set(value))


def run_metrics(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: MetricsRunConfig,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Compute metrics from normalized qrels and a retrieval_run artifact."""

    dataset = read_normalized_dataset_artifact(
        source_store,
        config.source_normalized_dataset_artifact_id,
    )
    retrieval_records = read_retrieval_run_artifact(
        source_store,
        config.source_retrieval_run_artifact_id,
    )
    data = build_metrics_run_data(
        dataset,
        retrieval_records,
        config,
        progress_reporter=progress_reporter,
    )
    return write_metrics_run_artifact(
        output_store,
        config.output_artifact_id,
        data,
        queries_per_shard=config.queries_per_shard,
        metadata=_build_manifest_metadata(config, data, source_store=source_store),
        dependencies=[
            ArtifactDependency(
                artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
                artifact_id=config.source_normalized_dataset_artifact_id,
            ),
            ArtifactDependency(
                artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
                artifact_id=config.source_retrieval_run_artifact_id,
            ),
        ],
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )


def build_metrics_run_fingerprint_sha256(
    source_store: ArtifactStore,
    config: MetricsRunConfig,
) -> str | None:
    """Return the expected metrics_run asset fingerprint for a config, if available."""

    components = _metrics_asset_fingerprint_components(config, source_store=source_store)
    if components is None:
        return None
    return build_asset_fingerprint(
        artifact_type=METRICS_RUN_ARTIFACT_TYPE,
        components=components,
    ).sha256


def build_metrics_run_data(
    dataset: NormalizedDataset,
    retrieval_records: list[RetrievalQueryResult],
    config: MetricsRunConfig,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> MetricsRunData:
    qrels_by_query: dict[str, dict[str, float]] = {}
    all_qrel_query_ids: set[str] = set()
    for qrel in dataset.qrels:
        all_qrel_query_ids.add(qrel.query_id)
        if qrel.relevance > 0:
            qrels_by_query.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    positive_query_ids = set(qrels_by_query)
    skipped_no_positive_qrels_query_count = len(all_qrel_query_ids - positive_query_ids)

    results_by_query = {record.query_id: record for record in retrieval_records}
    query_text_by_id = {query.query_id: query.text for query in dataset.queries}
    ignored_result_query_count = sum(
        1 for record in retrieval_records if record.query_id not in positive_query_ids
    )

    query_metrics: list[QueryMetricsRecord] = []
    missing_result_query_count = 0
    failed_retrieval_query_count = 0
    missing_doc_id_hit_count = 0
    duplicate_doc_hit_count = 0

    sorted_positive_query_ids = sorted(positive_query_ids)
    total_queries = len(sorted_positive_query_ids)
    report_progress(
        progress_reporter,
        stage="metrics_run",
        current=0,
        total=total_queries,
        message="Starting metrics queries",
        metadata=_progress_metadata(config),
    )

    for query_index, query_id in enumerate(sorted_positive_query_ids, start=1):
        relevant_docs = qrels_by_query[query_id]
        result = results_by_query.get(query_id)
        query_text = query_text_by_id.get(query_id, result.query_text if result else "")
        retrieval_error: str | None = None
        stats = ProjectionStats(
            input_hit_count=0,
            ranked_doc_count=0,
            missing_doc_id_hit_count=0,
            duplicate_doc_hit_count=0,
        )
        ranked_docs: list[RankedDoc] = []
        if result is None:
            missing_result_query_count += 1
        elif result.error is not None:
            failed_retrieval_query_count += 1
            retrieval_error = result.error
            query_text = result.query_text
        else:
            query_text = result.query_text
            ranked_docs, stats = project_retrieval_result_to_docs(result)

        missing_doc_id_hit_count += stats.missing_doc_id_hit_count
        duplicate_doc_hit_count += stats.duplicate_doc_hit_count
        metrics = compute_query_metrics(ranked_docs, relevant_docs, config.k_values)
        query_metrics.append(
            QueryMetricsRecord(
                query_id=query_id,
                query_text=query_text,
                retrieval_error=retrieval_error,
                ranked_docs=ranked_docs,
                relevant_docs=relevant_docs,
                metrics=metrics,
                projection_stats=stats,
            )
        )
        report_progress(
            progress_reporter,
            stage="metrics_run",
            current=query_index,
            total=total_queries,
            message="Computed query metrics",
            metadata={
                **_progress_metadata(config),
                "query_id": query_id,
                "missing_result_query_count": missing_result_query_count,
                "failed_retrieval_query_count": failed_retrieval_query_count,
            },
        )

    aggregate = aggregate_query_metrics(
        [record.metrics for record in query_metrics],
        config.k_values,
    )
    main_score = aggregate.get(MAIN_SCORE_METRIC, 0.0)
    metadata = {
        "missing_result_query_count": missing_result_query_count,
        "failed_retrieval_query_count": failed_retrieval_query_count,
        "ignored_result_query_count": ignored_result_query_count,
        "skipped_no_positive_qrels_query_count": skipped_no_positive_qrels_query_count,
        "missing_doc_id_hit_count": missing_doc_id_hit_count,
        "duplicate_doc_hit_count": duplicate_doc_hit_count,
    }
    return MetricsRunData(
        aggregate=aggregate,
        k_values=config.k_values,
        main_score=main_score,
        main_score_metric=MAIN_SCORE_METRIC,
        query_metrics=query_metrics,
        metadata=metadata,
    )


def _progress_metadata(config: MetricsRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "output_artifact_id": config.output_artifact_id,
            "source_normalized_dataset_artifact_id": (
                config.source_normalized_dataset_artifact_id
            ),
            "source_retrieval_run_artifact_id": config.source_retrieval_run_artifact_id,
            "k_values": config.k_values,
            "doc_aggregation": config.doc_aggregation,
            "doc_score": config.doc_score,
        }
    )
    return metadata


def _build_manifest_metadata(
    config: MetricsRunConfig,
    data: MetricsRunData,
    *,
    source_store: ArtifactStore | None = None,
) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(data.metadata)
    metadata.update(
        {
            "source_normalized_dataset_artifact_id": (
                config.source_normalized_dataset_artifact_id
            ),
            "source_retrieval_run_artifact_id": config.source_retrieval_run_artifact_id,
            "k_values": config.k_values,
            "doc_aggregation": config.doc_aggregation,
            "doc_score": config.doc_score,
            "main_score_metric": data.main_score_metric,
        }
    )
    add_asset_fingerprint_metadata(
        metadata,
        artifact_type="metrics_run",
        components=_metrics_asset_fingerprint_components(
            config,
            source_store=source_store,
        ),
    )
    return metadata


def _metrics_asset_fingerprint_components(
    config: MetricsRunConfig,
    *,
    source_store: ArtifactStore | None,
) -> dict[str, Any] | None:
    if source_store is None:
        return None
    try:
        normalized_manifest = source_store.read_manifest(
            NORMALIZED_DATASET_ARTIFACT_TYPE,
            config.source_normalized_dataset_artifact_id,
        )
        retrieval_manifest = source_store.read_manifest(
            RETRIEVAL_RUN_ARTIFACT_TYPE,
            config.source_retrieval_run_artifact_id,
        )
    except Exception:
        return None

    normalized_fingerprint = manifest_asset_fingerprint_sha256(normalized_manifest)
    retrieval_fingerprint = manifest_asset_fingerprint_sha256(retrieval_manifest)
    if normalized_fingerprint is None or retrieval_fingerprint is None:
        return None

    return metrics_run_fingerprint_components(
        normalized_dataset_fingerprint=normalized_fingerprint,
        retrieval_run_fingerprint=retrieval_fingerprint,
        metrics_source="sci-retrieval-eval",
        code_git_commit=config.code_git_sha or "unknown",
        metrics_entrypoint="eval_platform.metrics.runner.run_metrics",
        metric_params={
            "k_values": config.k_values,
            "main_metric": MAIN_SCORE_METRIC,
            "projection": {
                "from": "chunk",
                "to": "doc",
                "dedupe_policy": config.doc_aggregation,
                "score": config.doc_score,
            },
            "missing_query_policy": "zero",
        },
    )
