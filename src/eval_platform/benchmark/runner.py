"""Minimal benchmark run orchestration."""

from __future__ import annotations

from typing import Any

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.artifacts.types import BENCHMARK_RUN_ARTIFACT_TYPE
from eval_platform.assets import (
    add_asset_fingerprint_metadata,
    benchmark_run_fingerprint_components,
    build_asset_fingerprint,
    manifest_asset_fingerprint_sha256,
)
from eval_platform.benchmark.artifact import write_benchmark_run_artifact
from eval_platform.benchmark.schema import BenchmarkRunConfig, BenchmarkRunSummary
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.datasets import NORMALIZED_DATASET_ARTIFACT_TYPE
from eval_platform.embeddings import EmbeddingClient
from eval_platform.metrics import (
    METRICS_RUN_ARTIFACT_TYPE,
    read_metrics_run_artifact,
    run_metrics,
)
from eval_platform.retrieval import (
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RewriteClient,
    run_retrieval,
)


def run_benchmark(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: BenchmarkRunConfig,
    *,
    es_client: ElasticsearchRetrievalClient | None = None,
    milvus_client: MilvusRetrievalClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    rewrite_client: RewriteClient | None = None,
    rerank_client: RerankClient | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run retrieval then metrics and write a benchmark_run artifact."""

    report_progress(
        progress_reporter,
        stage="benchmark_run",
        current=0,
        total=3,
        message="Starting benchmark run",
        metadata=_progress_metadata(config),
    )
    retrieval_manifest = run_retrieval(
        source_store,
        output_store,
        config.retrieval,
        es_client=es_client,
        milvus_client=milvus_client,
        embedding_client=embedding_client,
        rewrite_client=rewrite_client,
        rerank_client=rerank_client,
        progress_reporter=progress_reporter,
    )
    report_progress(
        progress_reporter,
        stage="benchmark_run",
        current=1,
        total=3,
        message="Completed retrieval stage",
        metadata=_progress_metadata(config),
    )
    metrics_manifest = run_metrics(
        _BenchmarkReadStore(source_store, output_store),
        output_store,
        config.metrics,
        progress_reporter=progress_reporter,
    )
    report_progress(
        progress_reporter,
        stage="benchmark_run",
        current=2,
        total=3,
        message="Completed metrics stage",
        metadata=_progress_metadata(config),
    )
    return write_benchmark_run_from_existing_artifacts(
        output_store,
        config,
        retrieval_manifest=retrieval_manifest,
        metrics_manifest=metrics_manifest,
        progress_reporter=progress_reporter,
    )


def write_benchmark_run_from_existing_artifacts(
    output_store: ArtifactStore,
    config: BenchmarkRunConfig,
    *,
    retrieval_manifest: ArtifactManifest,
    metrics_manifest: ArtifactManifest,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Write a benchmark_run summary for already materialized retrieval/metrics runs."""

    metrics_data = read_metrics_run_artifact(output_store, config.metrics.output_artifact_id)
    summary = BenchmarkRunSummary(
        benchmark_run_artifact_id=config.output_artifact_id,
        setting_name=config.setting_name,
        retrieval_run_artifact_id=config.retrieval.output_artifact_id,
        metrics_run_artifact_id=config.metrics.output_artifact_id,
        source_normalized_dataset_artifact_id=config.source_normalized_dataset_artifact_id,
        main_score=metrics_data.main_score,
        main_score_metric=metrics_data.main_score_metric,
        aggregate_metrics=metrics_data.aggregate,
    )
    report_progress(
        progress_reporter,
        stage="benchmark_run",
        current=3,
        total=3,
        message="Writing benchmark run artifact",
        metadata={
            **_progress_metadata(config),
            "main_score": metrics_data.main_score,
            "main_score_metric": metrics_data.main_score_metric,
        },
    )
    return write_benchmark_run_artifact(
        output_store,
        config.output_artifact_id,
        summary,
        metadata=_build_manifest_metadata(config, retrieval_manifest, metrics_manifest),
        dependencies=[
            ArtifactDependency(
                artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
                artifact_id=config.source_normalized_dataset_artifact_id,
            ),
            ArtifactDependency(
                artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
                artifact_id=config.retrieval.output_artifact_id,
            ),
            ArtifactDependency(
                artifact_type=METRICS_RUN_ARTIFACT_TYPE,
                artifact_id=config.metrics.output_artifact_id,
            ),
        ],
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )


def build_benchmark_run_fingerprint_sha256(
    config: BenchmarkRunConfig,
    *,
    retrieval_manifest: ArtifactManifest,
    metrics_manifest: ArtifactManifest,
) -> str | None:
    """Return the expected benchmark_run asset fingerprint for existing child runs."""

    components = _benchmark_asset_fingerprint_components(
        config,
        retrieval_manifest=retrieval_manifest,
        metrics_manifest=metrics_manifest,
    )
    if components is None:
        return None
    return build_asset_fingerprint(
        artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
        components=components,
    ).sha256


def _progress_metadata(config: BenchmarkRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "output_artifact_id": config.output_artifact_id,
            "source_normalized_dataset_artifact_id": (
                config.source_normalized_dataset_artifact_id
            ),
            "setting_name": config.setting_name,
            "retrieval_run_artifact_id": config.retrieval.output_artifact_id,
            "metrics_run_artifact_id": config.metrics.output_artifact_id,
            "retrieval_mode": config.retrieval.retrieval_mode,
            "retrieval_execution_mode": config.retrieval.execution_mode,
            "query_limit": config.retrieval.query_limit,
        }
    )
    return metadata


def _build_manifest_metadata(
    config: BenchmarkRunConfig,
    retrieval_manifest: ArtifactManifest,
    metrics_manifest: ArtifactManifest,
) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "source_normalized_dataset_artifact_id": (
                config.source_normalized_dataset_artifact_id
            ),
            "retrieval_run_artifact_id": config.retrieval.output_artifact_id,
            "metrics_run_artifact_id": config.metrics.output_artifact_id,
            "setting_name": config.setting_name,
            "description": config.description,
            "tags": config.tags,
            "retrieval_mode": config.retrieval.retrieval_mode,
            "retrieval_execution_mode": config.retrieval.execution_mode,
            "retrieval_trace_mode": config.retrieval.trace_mode,
            "top_k": config.retrieval.top_k,
            "sub_queries": config.retrieval.sub_queries,
            "rewrite_enabled": config.retrieval.rewrite_enabled,
            "rerank_enabled": config.retrieval.rerank_enabled,
            "metrics_k_values": config.metrics.k_values,
            "doc_aggregation": config.metrics.doc_aggregation,
            "doc_score": config.metrics.doc_score,
            "main_score_metric": metrics_manifest.metadata.get("main_score_metric"),
            "main_score": metrics_manifest.metadata.get("main_score"),
            "retrieval_failed_query_count": retrieval_manifest.metadata.get(
                "failed_query_count"
            ),
            "metrics_evaluated_query_count": metrics_manifest.metadata.get(
                "evaluated_query_count"
            ),
        }
    )
    add_asset_fingerprint_metadata(
        metadata,
        artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
        components=_benchmark_asset_fingerprint_components(
            config,
            retrieval_manifest=retrieval_manifest,
            metrics_manifest=metrics_manifest,
        ),
    )
    return metadata


def _benchmark_asset_fingerprint_components(
    config: BenchmarkRunConfig,
    *,
    retrieval_manifest: ArtifactManifest,
    metrics_manifest: ArtifactManifest,
) -> dict[str, Any] | None:
    retrieval_fingerprint = manifest_asset_fingerprint_sha256(retrieval_manifest)
    metrics_fingerprint = manifest_asset_fingerprint_sha256(metrics_manifest)
    if retrieval_fingerprint is None or metrics_fingerprint is None:
        return None

    return benchmark_run_fingerprint_components(
        retrieval_run_fingerprint=retrieval_fingerprint,
        metrics_run_fingerprint=metrics_fingerprint,
        benchmark_source="sci-retrieval-eval",
        code_git_commit=config.code_git_sha or "unknown",
        benchmark_entrypoint="eval_platform.benchmark.runner.write_benchmark_run_from_existing_artifacts",
        setting_name=config.setting_name,
        benchmark_params={},
    )


class _BenchmarkReadStore(ArtifactStore):
    """Read normalized input from source_store and generated run artifacts from output_store."""

    def __init__(self, source_store: ArtifactStore, output_store: ArtifactStore) -> None:
        self._source_store = source_store
        self._output_store = output_store

    def _read_store(self, artifact_type: str) -> ArtifactStore:
        if artifact_type == NORMALIZED_DATASET_ARTIFACT_TYPE:
            return self._source_store
        return self._output_store

    def put_file(
        self,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
        data: bytes,
    ) -> None:
        raise RuntimeError("_BenchmarkReadStore is read-only")

    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        return self._read_store(artifact_type).get_file(artifact_type, artifact_id, relative_path)

    def exists(self, artifact_type: str, artifact_id: str, relative_path: str) -> bool:
        return self._read_store(artifact_type).exists(artifact_type, artifact_id, relative_path)

    def list_artifacts(self, artifact_type: str | None = None) -> list[tuple[str, str]]:
        if artifact_type == NORMALIZED_DATASET_ARTIFACT_TYPE:
            return self._source_store.list_artifacts(artifact_type)
        return self._output_store.list_artifacts(artifact_type)

    def write_manifest(
        self,
        artifact_type: str,
        artifact_id: str,
        manifest: ArtifactManifest,
    ) -> None:
        raise RuntimeError("_BenchmarkReadStore is read-only")

    def read_manifest(self, artifact_type: str, artifact_id: str) -> ArtifactManifest:
        return self._read_store(artifact_type).read_manifest(artifact_type, artifact_id)

    def mark_success(self, artifact_type: str, artifact_id: str) -> None:
        raise RuntimeError("_BenchmarkReadStore is read-only")

    def is_complete(self, artifact_type: str, artifact_id: str) -> bool:
        return self._read_store(artifact_type).is_complete(artifact_type, artifact_id)

    def artifact_uri(self, artifact_type: str, artifact_id: str) -> str:
        return self._read_store(artifact_type).artifact_uri(artifact_type, artifact_id)
