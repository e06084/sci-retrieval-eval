"""Experiment planning and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.artifacts.catalog import (
    ArtifactCatalogRecord,
    build_artifact_catalog_record,
    find_catalog_record_by_fingerprint,
    read_artifact_catalog,
    upsert_artifact_catalog_record,
)
from eval_platform.artifacts.types import (
    BENCHMARK_RUN_ARTIFACT_TYPE,
    METRICS_RUN_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RETRIEVAL_RUN_ARTIFACT_TYPE,
)
from eval_platform.assets import (
    benchmark_run_fingerprint_components,
    build_asset_fingerprint,
    manifest_asset_fingerprint_sha256,
    metrics_run_fingerprint_components,
)
from eval_platform.benchmark import (
    BenchmarkDatasetSpec,
    BenchmarkRunConfig,
    BenchmarkSuiteRunConfig,
    build_benchmark_run_config,
    read_benchmark_run_artifact,
    write_benchmark_run_from_existing_artifacts,
)
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.datasets import read_normalized_dataset_artifact
from eval_platform.defaults import DEFAULT_MAIN_SCORE_METRIC
from eval_platform.embeddings import EmbeddingClient
from eval_platform.experiments.artifact import write_experiment_run_artifact
from eval_platform.experiments.corpus_assets import (
    resolve_benchmark_datasets_from_corpus_assets,
)
from eval_platform.experiments.schema import (
    ExperimentItemPlan,
    ExperimentPlan,
    ExperimentRunConfig,
    ExperimentRunItemSummary,
    ExperimentRunSummary,
    ExperimentStagePlan,
)
from eval_platform.metrics import (
    MetricsRunConfig,
    build_metrics_run_fingerprint_sha256,
    run_metrics,
)
from eval_platform.retrieval import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RewriteClient,
    read_retrieval_run_artifact,
    build_retrieval_run_fingerprint_sha256,
    run_retrieval,
)


@dataclass(frozen=True)
class _ResolvedArtifact:
    artifact_id: str
    manifest: ArtifactManifest | None
    action: str
    expected_fingerprint: str | None
    generated_artifact_id: str
    reuse_reason: str | None = None


@dataclass(frozen=True)
class _ResolvedExperimentItem:
    dataset_key: str
    setting_key: str
    benchmark_config: BenchmarkRunConfig
    retrieval: _ResolvedArtifact
    metrics: _ResolvedArtifact
    benchmark: _ResolvedArtifact


@dataclass(frozen=True)
class _ResolvedExperiment:
    datasets: list[BenchmarkDatasetSpec]
    items: list[_ResolvedExperimentItem]
    corpus_asset_plan: dict[str, Any] | None = None


class ExperimentRunError(Exception):
    """Raised when experiment planning or execution fails."""


def plan_experiment(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: ExperimentRunConfig,
) -> ExperimentPlan:
    """Plan reuse/create actions for all dataset x setting benchmark items."""

    resolved = _resolve_experiment(source_store, output_store, config)
    return _build_plan(config, resolved)


def _build_plan(
    config: ExperimentRunConfig,
    resolved: _ResolvedExperiment,
) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_run_id=config.experiment_run_id,
        item_count=len(resolved.items),
        dataset_count=len(resolved.datasets),
        setting_count=len(config.settings),
        reuse_existing=config.reuse_existing,
        corpus_asset_plan=resolved.corpus_asset_plan,
        items=[_item_plan(item) for item in resolved.items],
    )


def run_experiment(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: ExperimentRunConfig,
    *,
    es_client: ElasticsearchRetrievalClient | None = None,
    milvus_client: MilvusRetrievalClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    rewrite_client: RewriteClient | None = None,
    rerank_client: RerankClient | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run only missing experiment items/stages and write an experiment_run artifact."""

    resolved = _resolve_experiment(source_store, output_store, config)
    initial_plan = _build_plan(config, resolved)
    total_items = len(resolved.items)
    report_progress(
        progress_reporter,
        stage="experiment_run",
        current=0,
        total=total_items,
        message="Starting experiment run",
        metadata={"experiment_run_id": config.experiment_run_id},
    )

    summaries: list[ExperimentRunItemSummary] = []
    dependencies: list[ArtifactDependency] = []
    read_store = _ExperimentReadStore(source_store, output_store)

    for item_index, item in enumerate(resolved.items, start=1):
        report_progress(
            progress_reporter,
            stage="experiment_run",
            current=item_index - 1,
            total=total_items,
            message="Starting experiment item",
            metadata=_item_progress_metadata(config, item),
        )
        benchmark_manifest = _materialize_experiment_item(
            source_store,
            output_store,
            read_store,
            item,
            es_client=es_client,
            milvus_client=milvus_client,
            embedding_client=embedding_client,
            rewrite_client=rewrite_client,
            rerank_client=rerank_client,
            progress_reporter=progress_reporter,
        )
        benchmark_summary = read_benchmark_run_artifact(
            output_store,
            benchmark_manifest.artifact_id,
        )
        if config.catalog_id is not None:
            for artifact_type, artifact_id in (
                (RETRIEVAL_RUN_ARTIFACT_TYPE, benchmark_summary.retrieval_run_artifact_id),
                (METRICS_RUN_ARTIFACT_TYPE, benchmark_summary.metrics_run_artifact_id),
                (BENCHMARK_RUN_ARTIFACT_TYPE, benchmark_summary.benchmark_run_artifact_id),
            ):
                _upsert_artifact_catalog_record_if_complete(
                    output_store,
                    artifact_type=artifact_type,
                    artifact_id=artifact_id,
                    catalog_id=config.catalog_id,
                    created_by=config.created_by,
                    code_git_sha=config.code_git_sha,
                )
        aggregate_metrics = dict(benchmark_summary.aggregate_metrics)
        aggregate_metrics.update(
            _compute_recall_inf_metrics(
                output_store,
                source_normalized_dataset_artifact_id=(
                    benchmark_summary.source_normalized_dataset_artifact_id
                ),
                retrieval_run_artifact_id=benchmark_summary.retrieval_run_artifact_id,
            )
        )
        summaries.append(
            ExperimentRunItemSummary(
                dataset_key=item.dataset_key,
                setting_key=item.setting_key,
                benchmark_run_artifact_id=benchmark_summary.benchmark_run_artifact_id,
                retrieval_run_artifact_id=benchmark_summary.retrieval_run_artifact_id,
                metrics_run_artifact_id=benchmark_summary.metrics_run_artifact_id,
                actions={
                    "retrieval_run": item.retrieval.action,
                    "metrics_run": item.metrics.action,
                    "benchmark_run": item.benchmark.action,
                },
                main_score=benchmark_summary.main_score,
                main_score_metric=benchmark_summary.main_score_metric,
                aggregate_metrics=aggregate_metrics,
            )
        )
        dependencies.append(
            ArtifactDependency(
                artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
                artifact_id=benchmark_summary.benchmark_run_artifact_id,
            )
        )
        report_progress(
            progress_reporter,
            stage="experiment_run",
            current=item_index,
            total=total_items,
            message="Completed experiment item",
            metadata={
                **_item_progress_metadata(config, item),
                "main_score": benchmark_summary.main_score,
                "main_score_metric": benchmark_summary.main_score_metric,
            },
        )

    summary = ExperimentRunSummary(
        experiment_run_id=config.experiment_run_id,
        item_count=len(summaries),
        dataset_count=len(resolved.datasets),
        setting_count=len(config.settings),
        items=summaries,
    )
    manifest = write_experiment_run_artifact(
        output_store,
        config.experiment_run_id,
        summary,
        metadata={
            **config.metadata,
            "experiment_run_id": config.experiment_run_id,
            "reuse_existing": config.reuse_existing,
            "plan": initial_plan.model_dump(mode="json"),
        },
        dependencies=dependencies,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )
    if config.catalog_id is not None:
        upsert_artifact_catalog_record(
            output_store,
            build_artifact_catalog_record(
                output_store,
                manifest.artifact_type,
                manifest.artifact_id,
            ),
            config.catalog_id,
            created_by=config.created_by,
            code_git_sha=config.code_git_sha,
        )
    return manifest


def _upsert_artifact_catalog_record_if_complete(
    store: ArtifactStore,
    *,
    artifact_type: str,
    artifact_id: str,
    catalog_id: str,
    created_by: str | None,
    code_git_sha: str | None,
) -> None:
    if _read_complete_manifest(store, artifact_type, artifact_id) is None:
        return
    upsert_artifact_catalog_record(
        store,
        build_artifact_catalog_record(store, artifact_type, artifact_id),
        catalog_id,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )


def _compute_recall_inf_metrics(
    store: ArtifactStore,
    *,
    source_normalized_dataset_artifact_id: str,
    retrieval_run_artifact_id: str,
) -> dict[str, float]:
    try:
        dataset = read_normalized_dataset_artifact(store, source_normalized_dataset_artifact_id)
        retrieval_records = read_retrieval_run_artifact(
            store,
            retrieval_run_artifact_id,
            include_trace=True,
        )
    except Exception:
        return {}

    qrels_by_query: dict[str, set[str]] = {}
    for qrel in dataset.qrels:
        if qrel.relevance <= 0:
            continue
        qrels_by_query.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    if not qrels_by_query:
        return {}

    records_by_query_id = {record.query_id: record for record in retrieval_records}
    es_inf: list[float] = []
    milvus_inf: list[float] = []
    rrf_inf: list[float] = []
    rrf_es_inf: list[float] = []
    rrf_milvus_inf: list[float] = []
    for query_id, relevant_docs in sorted(qrels_by_query.items()):
        record = records_by_query_id.get(query_id)
        if record is None or record.error:
            es_inf.append(0.0)
            milvus_inf.append(0.0)
            rrf_inf.append(0.0)
            rrf_es_inf.append(0.0)
            rrf_milvus_inf.append(0.0)
            continue
        trace = record.trace if isinstance(record.trace, dict) else {}
        es_docs = _trace_doc_ids(trace, "es_hits")
        milvus_docs = _trace_doc_ids(trace, "milvus_hits")
        rrf_docs = _trace_doc_ids(trace, "paper_capped_hits")
        if not rrf_docs:
            rrf_docs = _trace_doc_ids(trace, "fused_hits")
        if not rrf_docs:
            rrf_docs = _record_doc_ids(record.hits)
        es_inf.append(_recall_inf(es_docs, relevant_docs))
        milvus_inf.append(_recall_inf(milvus_docs, relevant_docs))
        rrf_inf.append(_recall_inf(rrf_docs, relevant_docs))
        rrf_es_inf.append(_recall_inf(rrf_docs & es_docs, relevant_docs))
        rrf_milvus_inf.append(_recall_inf(rrf_docs & milvus_docs, relevant_docs))
    return {
        "es_recall_at_inf": _mean(es_inf),
        "milvus_recall_at_inf": _mean(milvus_inf),
        "rrf_recall_at_inf": _mean(rrf_inf),
        "rrf_intersect_es_recall_at_inf": _mean(rrf_es_inf),
        "rrf_intersect_milvus_recall_at_inf": _mean(rrf_milvus_inf),
    }


def _trace_doc_ids(trace: dict[str, Any], key: str) -> set[str]:
    out: set[str] = set()
    top_level = trace.get(key)
    if isinstance(top_level, list):
        for hit in top_level:
            if isinstance(hit, dict):
                doc_id = _trace_hit_doc_id(hit)
                if doc_id:
                    out.add(doc_id)
    per_query = trace.get("per_query")
    if isinstance(per_query, list):
        for item in per_query:
            if not isinstance(item, dict):
                continue
            query_hits = item.get(key)
            if not isinstance(query_hits, list):
                continue
            for hit in query_hits:
                if isinstance(hit, dict):
                    doc_id = _trace_hit_doc_id(hit)
                    if doc_id:
                        out.add(doc_id)
    return out


def _record_doc_ids(hits: list[Any]) -> set[str]:
    out: set[str] = set()
    for hit in hits:
        doc_id = _trace_hit_doc_id(
            {
                "doc_id": getattr(hit, "doc_id", ""),
                "chunk_id": getattr(hit, "chunk_id", ""),
                "metadata": getattr(hit, "metadata", {}) or {},
            }
        )
        if doc_id:
            out.add(doc_id)
    return out


def _trace_hit_doc_id(hit: dict[str, Any]) -> str:
    doc_id = str(hit.get("doc_id") or "").strip()
    if doc_id:
        return doc_id
    metadata = hit.get("metadata") or {}
    if isinstance(metadata, dict):
        paper_id = metadata.get("paper_id")
        if isinstance(paper_id, str) and paper_id.strip():
            return paper_id.strip()
    return str(hit.get("chunk_id") or "").strip()


def _recall_inf(doc_ids: set[str], relevant_docs: set[str]) -> float:
    if not relevant_docs:
        return 0.0
    return len(doc_ids & relevant_docs) / len(relevant_docs)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _materialize_experiment_item(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    read_store: ArtifactStore,
    item: _ResolvedExperimentItem,
    *,
    es_client: ElasticsearchRetrievalClient | None,
    milvus_client: MilvusRetrievalClient | None,
    embedding_client: EmbeddingClient | None,
    rewrite_client: RewriteClient | None,
    rerank_client: RerankClient | None,
    progress_reporter: ProgressReporter | None,
) -> ArtifactManifest:
    if item.benchmark.action == "reuse" and item.benchmark.manifest is not None:
        return item.benchmark.manifest

    if item.retrieval.action == "create":
        retrieval_manifest = run_retrieval(
            source_store,
            output_store,
            item.benchmark_config.retrieval,
            es_client=es_client,
            milvus_client=milvus_client,
            embedding_client=embedding_client,
            rewrite_client=rewrite_client,
            rerank_client=rerank_client,
            progress_reporter=progress_reporter,
        )
    elif item.retrieval.manifest is not None:
        retrieval_manifest = item.retrieval.manifest
    else:
        retrieval_manifest = output_store.read_manifest(
            RETRIEVAL_RUN_ARTIFACT_TYPE,
            item.retrieval.artifact_id,
        )

    if item.metrics.action == "create":
        metrics_manifest = run_metrics(
            read_store,
            output_store,
            item.benchmark_config.metrics,
            progress_reporter=progress_reporter,
        )
    elif item.metrics.manifest is not None:
        metrics_manifest = item.metrics.manifest
    else:
        metrics_manifest = output_store.read_manifest(
            METRICS_RUN_ARTIFACT_TYPE,
            item.metrics.artifact_id,
        )

    return write_benchmark_run_from_existing_artifacts(
        output_store,
        item.benchmark_config,
        retrieval_manifest=retrieval_manifest,
        metrics_manifest=metrics_manifest,
        progress_reporter=progress_reporter,
    )


def _resolve_experiment(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: ExperimentRunConfig,
) -> _ResolvedExperiment:
    datasets, corpus_asset_plan = _resolve_experiment_datasets(source_store, config)
    suite_config = _benchmark_suite_config(config, datasets)
    read_store = _ExperimentReadStore(source_store, output_store)
    catalog_records = (
        read_artifact_catalog(output_store, config.catalog_id)
        if config.catalog_id is not None
        else None
    )
    items: list[_ResolvedExperimentItem] = []

    for dataset in datasets:
        for setting in config.settings:
            benchmark_config = build_benchmark_run_config(suite_config, dataset, setting)
            retrieval_fingerprint = build_retrieval_run_fingerprint_sha256(
                source_store,
                benchmark_config.retrieval,
            )
            retrieval = _resolve_artifact(
                output_store,
                artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
                generated_artifact_id=benchmark_config.retrieval.output_artifact_id,
                expected_fingerprint=retrieval_fingerprint,
                reuse_existing=config.reuse_existing,
                catalog_records=catalog_records,
            )
            benchmark_config = _with_retrieval_artifact(benchmark_config, retrieval.artifact_id)

            metrics_fingerprint = _build_metrics_fingerprint_from_expected_retrieval(
                read_store,
                benchmark_config.metrics,
                retrieval_fingerprint=retrieval_fingerprint,
            )
            metrics = _resolve_artifact(
                output_store,
                artifact_type=METRICS_RUN_ARTIFACT_TYPE,
                generated_artifact_id=benchmark_config.metrics.output_artifact_id,
                expected_fingerprint=metrics_fingerprint,
                reuse_existing=config.reuse_existing,
                catalog_records=catalog_records,
            )
            benchmark_config = _with_metrics_artifact(benchmark_config, metrics.artifact_id)

            benchmark_fingerprint = _build_benchmark_fingerprint_from_expected_children(
                benchmark_config,
                retrieval_fingerprint=retrieval_fingerprint,
                metrics_fingerprint=metrics_fingerprint,
            )
            benchmark = _resolve_artifact(
                output_store,
                artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
                generated_artifact_id=benchmark_config.output_artifact_id,
                expected_fingerprint=benchmark_fingerprint,
                reuse_existing=config.reuse_existing,
                catalog_records=catalog_records,
            )
            benchmark_config = benchmark_config.model_copy(
                update={"output_artifact_id": benchmark.artifact_id}
            )
            benchmark_config, retrieval, metrics = _align_reused_benchmark_children(
                output_store,
                benchmark_config,
                retrieval=retrieval,
                metrics=metrics,
                benchmark=benchmark,
            )

            items.append(
                _ResolvedExperimentItem(
                    dataset_key=dataset.dataset_key,
                    setting_key=setting.setting_key,
                    benchmark_config=benchmark_config,
                    retrieval=retrieval,
                    metrics=metrics,
                    benchmark=benchmark,
                )
            )
    return _ResolvedExperiment(
        datasets=datasets,
        items=items,
        corpus_asset_plan=corpus_asset_plan,
    )


def _align_reused_benchmark_children(
    store: ArtifactStore,
    config: BenchmarkRunConfig,
    *,
    retrieval: _ResolvedArtifact,
    metrics: _ResolvedArtifact,
    benchmark: _ResolvedArtifact,
) -> tuple[BenchmarkRunConfig, _ResolvedArtifact, _ResolvedArtifact]:
    if benchmark.action != "reuse":
        return config, retrieval, metrics
    try:
        summary = read_benchmark_run_artifact(store, benchmark.artifact_id)
    except Exception:
        return config, retrieval, metrics

    resolved_retrieval = _ResolvedArtifact(
        artifact_id=summary.retrieval_run_artifact_id,
        manifest=_read_complete_manifest(
            store,
            RETRIEVAL_RUN_ARTIFACT_TYPE,
            summary.retrieval_run_artifact_id,
        ),
        action="reuse",
        expected_fingerprint=retrieval.expected_fingerprint,
        generated_artifact_id=retrieval.generated_artifact_id,
        reuse_reason="benchmark_run_summary",
    )
    resolved_metrics = _ResolvedArtifact(
        artifact_id=summary.metrics_run_artifact_id,
        manifest=_read_complete_manifest(
            store,
            METRICS_RUN_ARTIFACT_TYPE,
            summary.metrics_run_artifact_id,
        ),
        action="reuse",
        expected_fingerprint=metrics.expected_fingerprint,
        generated_artifact_id=metrics.generated_artifact_id,
        reuse_reason="benchmark_run_summary",
    )
    aligned_config = _with_retrieval_artifact(config, resolved_retrieval.artifact_id)
    aligned_config = _with_metrics_artifact(aligned_config, resolved_metrics.artifact_id)
    return aligned_config, resolved_retrieval, resolved_metrics


def _resolve_experiment_datasets(
    source_store: ArtifactStore,
    config: ExperimentRunConfig,
) -> tuple[list[BenchmarkDatasetSpec], dict[str, Any] | None]:
    if config.datasets:
        return config.datasets, None
    if config.corpus_assets is None:
        raise ExperimentRunError("exactly one of datasets or corpus_assets must be provided")
    datasets, corpus_asset_plan = resolve_benchmark_datasets_from_corpus_assets(
        source_store,
        config.corpus_assets,
    )
    return datasets, corpus_asset_plan


def _benchmark_suite_config(
    config: ExperimentRunConfig,
    datasets: list[BenchmarkDatasetSpec],
) -> BenchmarkSuiteRunConfig:
    return BenchmarkSuiteRunConfig(
        suite_run_id=config.experiment_run_id,
        datasets=datasets,
        settings=config.settings,
        metrics_k_values=config.metrics_k_values,
        query_limit=config.query_limit,
        queries_per_shard=config.queries_per_shard,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        metadata={
            **config.metadata,
            "experiment_run_id": config.experiment_run_id,
        },
    )


def _resolve_artifact(
    store: ArtifactStore,
    *,
    artifact_type: str,
    generated_artifact_id: str,
    expected_fingerprint: str | None,
    reuse_existing: bool,
    catalog_records: list[ArtifactCatalogRecord] | None,
) -> _ResolvedArtifact:
    if expected_fingerprint is not None and reuse_existing:
        if catalog_records is not None:
            record = find_catalog_record_by_fingerprint(
                catalog_records,
                artifact_type=artifact_type,
                asset_fingerprint_sha256=expected_fingerprint,
            )
            if record is not None:
                manifest = _read_complete_manifest(
                    store,
                    artifact_type,
                    record.artifact_id,
                )
                if manifest is not None:
                    return _ResolvedArtifact(
                        artifact_id=manifest.artifact_id,
                        manifest=manifest,
                        action="reuse",
                        expected_fingerprint=expected_fingerprint,
                        generated_artifact_id=generated_artifact_id,
                        reuse_reason="catalog_asset_fingerprint",
                    )

        manifest = _find_complete_manifest_by_fingerprint(
            store,
            artifact_type,
            expected_fingerprint,
        )
        if manifest is not None:
            return _ResolvedArtifact(
                artifact_id=manifest.artifact_id,
                manifest=manifest,
                action="reuse",
                expected_fingerprint=expected_fingerprint,
                generated_artifact_id=generated_artifact_id,
                reuse_reason="asset_fingerprint",
            )

    exact_manifest = _read_complete_manifest(store, artifact_type, generated_artifact_id)
    if exact_manifest is not None:
        exact_fingerprint = manifest_asset_fingerprint_sha256(exact_manifest)
        if expected_fingerprint is None or exact_fingerprint == expected_fingerprint:
            return _ResolvedArtifact(
                artifact_id=generated_artifact_id,
                manifest=exact_manifest,
                action="reuse",
                expected_fingerprint=expected_fingerprint,
                generated_artifact_id=generated_artifact_id,
                reuse_reason="exact_artifact_id",
            )

    return _ResolvedArtifact(
        artifact_id=generated_artifact_id,
        manifest=None,
        action="create",
        expected_fingerprint=expected_fingerprint,
        generated_artifact_id=generated_artifact_id,
    )


def _find_complete_manifest_by_fingerprint(
    store: ArtifactStore,
    artifact_type: str,
    expected_fingerprint: str,
) -> ArtifactManifest | None:
    matches: list[ArtifactManifest] = []
    for _current_type, artifact_id in store.list_artifacts(artifact_type):
        manifest = _read_complete_manifest(store, artifact_type, artifact_id)
        if manifest is None:
            continue
        if manifest_asset_fingerprint_sha256(manifest) == expected_fingerprint:
            matches.append(manifest)
    if not matches:
        return None
    return sorted(matches, key=lambda item: (item.created_at, item.artifact_id), reverse=True)[0]


def _read_complete_manifest(
    store: ArtifactStore,
    artifact_type: str,
    artifact_id: str,
) -> ArtifactManifest | None:
    try:
        if not store.is_complete(artifact_type, artifact_id):
            return None
        return store.read_manifest(artifact_type, artifact_id)
    except Exception:
        return None


def _build_metrics_fingerprint_from_expected_retrieval(
    store: ArtifactStore,
    config: MetricsRunConfig,
    *,
    retrieval_fingerprint: str | None,
) -> str | None:
    if retrieval_fingerprint is None:
        return build_metrics_run_fingerprint_sha256(store, config)
    try:
        normalized_manifest = store.read_manifest(
            NORMALIZED_DATASET_ARTIFACT_TYPE,
            config.source_normalized_dataset_artifact_id,
        )
    except Exception:
        return build_metrics_run_fingerprint_sha256(store, config)
    normalized_fingerprint = manifest_asset_fingerprint_sha256(normalized_manifest)
    if normalized_fingerprint is None:
        return None
    components = metrics_run_fingerprint_components(
        normalized_dataset_fingerprint=normalized_fingerprint,
        retrieval_run_fingerprint=retrieval_fingerprint,
        metrics_source="sci-retrieval-eval",
        code_git_commit=config.code_git_sha or "unknown",
        metrics_entrypoint="eval_platform.metrics.runner.run_metrics",
        metric_params={
            "k_values": config.k_values,
            "main_metric": DEFAULT_MAIN_SCORE_METRIC,
            "projection": {
                "from": "chunk",
                "to": "doc",
                "dedupe_policy": config.doc_aggregation,
                "score": config.doc_score,
            },
            "missing_query_policy": "zero",
        },
    )
    return build_asset_fingerprint(
        artifact_type=METRICS_RUN_ARTIFACT_TYPE,
        components=components,
    ).sha256


def _build_benchmark_fingerprint_from_expected_children(
    config: BenchmarkRunConfig,
    *,
    retrieval_fingerprint: str | None,
    metrics_fingerprint: str | None,
) -> str | None:
    if retrieval_fingerprint is None or metrics_fingerprint is None:
        return None
    components = benchmark_run_fingerprint_components(
        retrieval_run_fingerprint=retrieval_fingerprint,
        metrics_run_fingerprint=metrics_fingerprint,
        benchmark_source="sci-retrieval-eval",
        code_git_commit=config.code_git_sha or "unknown",
        benchmark_entrypoint="eval_platform.benchmark.runner.write_benchmark_run_from_existing_artifacts",
        setting_name=config.setting_name,
        benchmark_params={},
    )
    return build_asset_fingerprint(
        artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
        components=components,
    ).sha256

def _with_retrieval_artifact(
    config: BenchmarkRunConfig,
    retrieval_artifact_id: str,
) -> BenchmarkRunConfig:
    retrieval = config.retrieval.model_copy(update={"output_artifact_id": retrieval_artifact_id})
    metrics = config.metrics.model_copy(
        update={"source_retrieval_run_artifact_id": retrieval_artifact_id}
    )
    return config.model_copy(update={"retrieval": retrieval, "metrics": metrics})


def _with_metrics_artifact(
    config: BenchmarkRunConfig,
    metrics_artifact_id: str,
) -> BenchmarkRunConfig:
    metrics = config.metrics.model_copy(update={"output_artifact_id": metrics_artifact_id})
    return config.model_copy(update={"metrics": metrics})


def _item_plan(item: _ResolvedExperimentItem) -> ExperimentItemPlan:
    return ExperimentItemPlan(
        dataset_key=item.dataset_key,
        setting_key=item.setting_key,
        retrieval=_stage_plan("retrieval_run", RETRIEVAL_RUN_ARTIFACT_TYPE, item.retrieval),
        metrics=_stage_plan("metrics_run", METRICS_RUN_ARTIFACT_TYPE, item.metrics),
        benchmark=_stage_plan("benchmark_run", BENCHMARK_RUN_ARTIFACT_TYPE, item.benchmark),
    )


def _stage_plan(
    stage: str,
    artifact_type: str,
    resolved: _ResolvedArtifact,
) -> ExperimentStagePlan:
    return ExperimentStagePlan(
        stage=stage,  # type: ignore[arg-type]
        action=resolved.action,  # type: ignore[arg-type]
        artifact_type=artifact_type,
        artifact_id=resolved.artifact_id,
        generated_artifact_id=resolved.generated_artifact_id,
        asset_fingerprint_sha256=resolved.expected_fingerprint,
        reuse_reason=resolved.reuse_reason,
    )


def _item_progress_metadata(
    config: ExperimentRunConfig,
    item: _ResolvedExperimentItem,
) -> dict[str, Any]:
    return {
        "experiment_run_id": config.experiment_run_id,
        "dataset_key": item.dataset_key,
        "setting_key": item.setting_key,
        "benchmark_run_artifact_id": item.benchmark.artifact_id,
        "retrieval_run_artifact_id": item.retrieval.artifact_id,
        "metrics_run_artifact_id": item.metrics.artifact_id,
        "actions": {
            "retrieval_run": item.retrieval.action,
            "metrics_run": item.metrics.action,
            "benchmark_run": item.benchmark.action,
        },
    }


class _ExperimentReadStore(ArtifactStore):
    """Read normalized inputs from source_store and generated runs from output_store."""

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
        raise RuntimeError("_ExperimentReadStore is read-only")

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
        raise RuntimeError("_ExperimentReadStore is read-only")

    def read_manifest(self, artifact_type: str, artifact_id: str) -> ArtifactManifest:
        return self._read_store(artifact_type).read_manifest(artifact_type, artifact_id)

    def mark_success(self, artifact_type: str, artifact_id: str) -> None:
        raise RuntimeError("_ExperimentReadStore is read-only")

    def is_complete(self, artifact_type: str, artifact_id: str) -> bool:
        return self._read_store(artifact_type).is_complete(artifact_type, artifact_id)

    def artifact_uri(self, artifact_type: str, artifact_id: str) -> str:
        return self._read_store(artifact_type).artifact_uri(artifact_type, artifact_id)
