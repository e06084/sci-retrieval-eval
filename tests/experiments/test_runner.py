"""Tests for experiment planning and reuse execution."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactCatalogRecord,
    ArtifactDependency,
    ArtifactManifest,
    LocalArtifactStore,
    read_artifact_catalog,
)
from eval_platform.artifacts.types import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    EMBEDDINGS_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
)
from eval_platform.benchmark import BenchmarkDatasetSpec, settings_for_selection
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.experiments import (
    ExperimentCorpusAssetConfig,
    ExperimentCorpusAssetResolutionError,
    ExperimentRunConfig,
    plan_experiment,
    read_experiment_run_artifact,
    run_experiment,
)
from eval_platform.retrieval import RetrievalHit


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.enrich_calls: list[list[str]] = []

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        self.search_calls.append((index_name, query, top_k))
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="chunk text",
                score=1.0,
                recall_source="es",
            )
        ]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        self.enrich_calls.append([hit.chunk_id for hit in hits])
        return list(hits)


def test_experiment_run_config_uses_default_recall_focus() -> None:
    config = ExperimentRunConfig(
        experiment_run_id="exp-defaults",
        datasets=[
            BenchmarkDatasetSpec(
                dataset_key="ds1",
                normalized_dataset_artifact_id="normalized-ds1",
                elasticsearch_index_artifact_id="ds-es",
                milvus_collection_artifact_id="ds-milvus",
                index_name="ds-index",
                collection_name="ds-collection",
            )
        ],
        settings=settings_for_selection("E2-es"),
    )

    assert config.metrics_k_values == [5, 10, 20]


def test_run_experiment_creates_missing_then_reuses_by_catalog(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    _write_small_dataset_and_assets(store)
    es_client = FakeElasticsearchClient()

    first_manifest = run_experiment(
        store,
        store,
        _experiment_config("exp-a", catalog_id="default"),
        es_client=es_client,
    )
    first_summary = read_experiment_run_artifact(store, "exp-a")
    catalog_records = read_artifact_catalog(store, "default")

    assert first_manifest.metadata["stage"] == "experiment_run"
    assert first_summary.item_count == 1
    assert first_summary.items[0].actions == {
        "retrieval_run": "create",
        "metrics_run": "create",
        "benchmark_run": "create",
    }
    assert es_client.search_calls == [("ds-index", "alpha", 100)]
    assert {
        (record.artifact_type, record.artifact_id)
        for record in catalog_records
        if isinstance(record, ArtifactCatalogRecord)
    } >= {
        ("retrieval_run", "exp-a__ds1__E2-es__retrieval"),
        ("metrics_run", "exp-a__ds1__E2-es__metrics"),
        ("benchmark_run", "exp-a__ds1__E2-es__benchmark"),
        ("experiment_run", "exp-a"),
    }

    second_es_client = FakeElasticsearchClient()
    second_manifest = run_experiment(
        store,
        store,
        _experiment_config("exp-b", catalog_id="default"),
        es_client=second_es_client,
    )
    second_summary = read_experiment_run_artifact(store, "exp-b")

    assert second_manifest.artifact_id == "exp-b"
    assert second_es_client.search_calls == []
    assert second_summary.items[0].actions == {
        "retrieval_run": "reuse",
        "metrics_run": "reuse",
        "benchmark_run": "reuse",
    }
    assert (
        second_summary.items[0].benchmark_run_artifact_id
        == "exp-a__ds1__E2-es__benchmark"
    )


def test_run_experiment_reuses_benchmark_without_recreating_child_stages(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    _write_small_dataset_and_assets(store)
    first_es_client = FakeElasticsearchClient()
    run_experiment(
        store,
        store,
        _experiment_config("exp-a", catalog_id="default"),
        es_client=first_es_client,
    )
    shutil.rmtree(store.artifact_dir("retrieval_run", "exp-a__ds1__E2-es__retrieval"))
    shutil.rmtree(store.artifact_dir("metrics_run", "exp-a__ds1__E2-es__metrics"))

    second_es_client = FakeElasticsearchClient()
    second_manifest = run_experiment(
        store,
        store,
        _experiment_config("exp-b", catalog_id="default"),
        es_client=second_es_client,
    )
    second_summary = read_experiment_run_artifact(store, second_manifest.artifact_id)

    assert second_es_client.search_calls == []
    assert second_summary.items[0].benchmark_run_artifact_id == (
        "exp-a__ds1__E2-es__benchmark"
    )
    assert second_summary.items[0].actions == {
        "retrieval_run": "reuse",
        "metrics_run": "reuse",
        "benchmark_run": "reuse",
    }


def test_run_experiment_creates_missing_from_corpus_asset_selection(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    run_id = "small_fp_case"
    _write_small_ifir_corpus_asset_chain(store, run_id)
    es_client = FakeElasticsearchClient()

    manifest = run_experiment(
        store,
        store,
        ExperimentRunConfig(
            experiment_run_id="exp-from-corpus-assets",
            corpus_assets=ExperimentCorpusAssetConfig(
                dataset_selection="IFIRNFCorpus",
                corpus_run_id=run_id,
                bucket="scibase-service",
            ),
            settings=settings_for_selection("E2-es"),
            metrics_k_values=[1, 10],
            code_git_sha="abc123",
        ),
        es_client=es_client,
    )
    summary = read_experiment_run_artifact(store, "exp-from-corpus-assets")

    assert manifest.metadata["plan"]["corpus_asset_plan"] is not None
    assert summary.item_count == 1
    assert summary.items[0].actions == {
        "retrieval_run": "create",
        "metrics_run": "create",
        "benchmark_run": "create",
    }
    assert es_client.search_calls == [
        (f"ifir_nfcorpus_{run_id}_es", "alpha", 100)
    ]


def test_plan_experiment_with_real_five_dataset_asset_case(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    datasets = _real_five_dataset_asset_specs()
    for dataset in datasets:
        _write_asset_manifest(
            store,
            "normalized_dataset",
            dataset.normalized_dataset_artifact_id,
            {
                "asset_fingerprint_sha256": f"fp-{dataset.dataset_key}-normalized",
                "task_name": dataset.task_name,
            },
        )
        _write_asset_manifest(
            store,
            "elasticsearch_index",
            dataset.elasticsearch_index_artifact_id,
            {
                "asset_fingerprint_sha256": f"fp-{dataset.dataset_key}-es",
                "index_name": dataset.index_name,
            },
        )
        _write_asset_manifest(
            store,
            "milvus_collection",
            dataset.milvus_collection_artifact_id,
            {
                "asset_fingerprint_sha256": f"fp-{dataset.dataset_key}-milvus",
                "collection_name": dataset.collection_name,
            },
        )

    plan = plan_experiment(
        store,
        store,
        ExperimentRunConfig(
            experiment_run_id="e1_e4_bge_m3_fp_20260530_experiment",
            datasets=datasets,
            settings=settings_for_selection(),
            query_limit=3,
            metrics_k_values=[1, 10, 100],
            code_git_sha="9b0c1dc2e7428b397385f4a9c03dc1b1c276ccb7",
        ),
    )

    assert plan.dataset_count == 5
    assert plan.setting_count == 4
    assert plan.item_count == 20
    assert {item.setting_key for item in plan.items} == {
        "E1-milvus",
        "E2-es",
        "E3-hybrid",
        "E4-hybrid-rerank",
    }
    assert all(item.retrieval.action == "create" for item in plan.items)
    assert all(item.metrics.action == "create" for item in plan.items)
    assert all(item.benchmark.action == "create" for item in plan.items)
    assert plan.items[0].retrieval.artifact_id.startswith(
        "e1_e4_bge_m3_fp_20260530_experiment__"
    )


def test_plan_experiment_resolves_real_five_dataset_corpus_assets(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    run_id = "e1_e4_bge_m3_fp_20260530"
    for slug, task_name in _real_five_dataset_rows():
        _write_real_corpus_asset_chain(store, slug=slug, task_name=task_name, run_id=run_id)

    plan = plan_experiment(
        store,
        store,
        ExperimentRunConfig(
            experiment_run_id=f"{run_id}_experiment",
            corpus_assets=ExperimentCorpusAssetConfig(
                dataset_selection="all",
                corpus_run_id=run_id,
                bucket="scibase-service",
                raw_prefix="sciverse_benchmark/raw",
                s3_prefix="sciverse_benchmark/assets",
            ),
            settings=settings_for_selection(),
            query_limit=3,
            metrics_k_values=[1, 10, 100],
            code_git_sha="9b0c1dc2e7428b397385f4a9c03dc1b1c276ccb7",
        ),
    )

    assert plan.dataset_count == 5
    assert plan.setting_count == 4
    assert plan.item_count == 20
    assert plan.corpus_asset_plan is not None
    ifir_plan = plan.corpus_asset_plan["datasets"]["IFIRNFCorpus"]
    assert ifir_plan["resolved_artifact_ids"][NORMALIZED_DATASET_ARTIFACT_TYPE] == (
        f"ifir_nfcorpus_{run_id}_normalized"
    )
    assert ifir_plan["resolved_resource_names"][ELASTICSEARCH_INDEX_ARTIFACT_TYPE] == (
        f"ifir_nfcorpus_{run_id}_es"
    )
    required_steps = {
        step["artifact_type"]: step["action"]
        for step in ifir_plan["steps"]
        if step["artifact_type"]
        in {
            NORMALIZED_DATASET_ARTIFACT_TYPE,
            ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
            MILVUS_COLLECTION_ARTIFACT_TYPE,
        }
    }
    assert required_steps == {
        NORMALIZED_DATASET_ARTIFACT_TYPE: "reuse",
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE: "reuse",
        MILVUS_COLLECTION_ARTIFACT_TYPE: "reuse",
    }
    assert {item.dataset_key for item in plan.items} == {
        "ifir_nfcorpus",
        "nfcorpus",
        "ifir_scifact",
        "scifact",
        "litsearch",
    }
    assert all(item.retrieval.action == "create" for item in plan.items)


def test_plan_experiment_from_corpus_assets_fails_when_required_assets_missing(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    run_id = "e1_e4_bge_m3_fp_20260530"
    raw_id = f"ifir_nfcorpus_{run_id}_raw"
    normalized_id = f"ifir_nfcorpus_{run_id}_normalized"
    _write_asset_manifest(
        store,
        RAW_DATASET_ARTIFACT_TYPE,
        raw_id,
        {
            "asset_fingerprint_sha256": "fp-ifir_nfcorpus-raw",
            "task_name": "IFIRNFCorpus",
        },
    )
    _write_asset_manifest(
        store,
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        normalized_id,
        {
            "asset_fingerprint_sha256": "fp-ifir_nfcorpus-normalized",
            "task_name": "IFIRNFCorpus",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=RAW_DATASET_ARTIFACT_TYPE,
                artifact_id=raw_id,
            )
        ],
    )

    with pytest.raises(
        ExperimentCorpusAssetResolutionError,
        match="Corpus assets are not complete",
    ):
        plan_experiment(
            store,
            store,
            ExperimentRunConfig(
                experiment_run_id=f"{run_id}_experiment",
                corpus_assets=ExperimentCorpusAssetConfig(
                    dataset_selection="IFIRNFCorpus",
                    corpus_run_id=run_id,
                    bucket="scibase-service",
                ),
                settings=settings_for_selection("E2-es"),
                code_git_sha="9b0c1dc2e7428b397385f4a9c03dc1b1c276ccb7",
            ),
        )


def _experiment_config(
    experiment_run_id: str,
    *,
    catalog_id: str | None = None,
) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        experiment_run_id=experiment_run_id,
        datasets=[
            BenchmarkDatasetSpec(
                dataset_key="ds1",
                task_name="Demo",
                normalized_dataset_artifact_id="normalized-ds1",
                elasticsearch_index_artifact_id="ds-es",
                milvus_collection_artifact_id="ds-milvus",
                index_name="ds-index",
                collection_name="ds-collection",
            )
        ],
        settings=settings_for_selection("E2-es"),
        metrics_k_values=[1, 10],
        catalog_id=catalog_id,
        code_git_sha="abc123",
    )


def _write_small_dataset_and_assets(store: LocalArtifactStore) -> None:
    write_normalized_dataset_artifact(
        store,
        "normalized-ds1",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="doc text")],
            queries=[QueryRecord(query_id="q-1", text="alpha")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
        ),
        metadata={
            "raw_dataset_asset_fingerprint_sha256": "fp-raw",
            "normalizer_name": "demo-normalizer",
            "normalizer_version": "1",
            "normalized_schema_version": "1",
        },
    )
    _write_asset_manifest(
        store,
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        "ds-es",
        {
            "asset_fingerprint_sha256": "fp-es",
            "index_name": "ds-index",
        },
    )
    _write_asset_manifest(
        store,
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        "ds-milvus",
        {
            "asset_fingerprint_sha256": "fp-milvus",
            "collection_name": "ds-collection",
        },
    )


def _write_small_ifir_corpus_asset_chain(
    store: LocalArtifactStore,
    run_id: str,
) -> None:
    slug = "ifir_nfcorpus"
    task_name = "IFIRNFCorpus"
    raw_id = f"{slug}_{run_id}_raw"
    normalized_id = f"{slug}_{run_id}_normalized"
    chunks_id = f"{slug}_{run_id}_chunks"
    embeddings_id = f"{slug}_{run_id}_embeddings"
    es_id = f"{slug}_{run_id}_es_index"
    milvus_id = f"{slug}_{run_id}_milvus_collection"
    _write_asset_manifest(
        store,
        RAW_DATASET_ARTIFACT_TYPE,
        raw_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-raw",
            "dataset_slug": slug,
            "task_name": task_name,
        },
    )
    write_normalized_dataset_artifact(
        store,
        normalized_id,
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="doc text")],
            queries=[QueryRecord(query_id="q-1", text="alpha")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
        ),
        metadata={
            "dataset_slug": slug,
            "task_name": task_name,
            "raw_dataset_asset_fingerprint_sha256": f"fp-{slug}-raw",
            "normalizer_name": "demo-normalizer",
            "normalizer_version": "1",
            "normalized_schema_version": "1",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=RAW_DATASET_ARTIFACT_TYPE,
                artifact_id=raw_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        CHUNKED_CORPUS_ARTIFACT_TYPE,
        chunks_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-chunks",
            "dataset_slug": slug,
            "task_name": task_name,
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
                artifact_id=normalized_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        EMBEDDINGS_ARTIFACT_TYPE,
        embeddings_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-embeddings",
            "dataset_slug": slug,
            "task_name": task_name,
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        es_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-es",
            "dataset_slug": slug,
            "task_name": task_name,
            "index_name": f"{slug}_{run_id}_es",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        milvus_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-milvus",
            "dataset_slug": slug,
            "task_name": task_name,
            "collection_name": f"{slug}_{run_id}_milvus",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            ),
            ArtifactDependency(
                artifact_type=EMBEDDINGS_ARTIFACT_TYPE,
                artifact_id=embeddings_id,
            ),
        ],
    )


def _write_asset_manifest(
    store: LocalArtifactStore,
    artifact_type: str,
    artifact_id: str,
    metadata: dict[str, object],
    dependencies: list[ArtifactDependency] | None = None,
) -> None:
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_at=datetime.now(UTC),
        metadata=metadata,
        dependencies=dependencies or [],
    )
    store.write_manifest(artifact_type, artifact_id, manifest)
    store.mark_success(artifact_type, artifact_id)


def _write_real_corpus_asset_chain(
    store: LocalArtifactStore,
    *,
    slug: str,
    task_name: str,
    run_id: str,
) -> None:
    raw_id = f"{slug}_{run_id}_raw"
    normalized_id = f"{slug}_{run_id}_normalized"
    chunks_id = f"{slug}_{run_id}_chunks"
    embeddings_id = f"{slug}_{run_id}_embeddings"
    es_id = f"{slug}_{run_id}_es_index"
    milvus_id = f"{slug}_{run_id}_milvus_collection"
    _write_asset_manifest(
        store,
        RAW_DATASET_ARTIFACT_TYPE,
        raw_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-raw",
            "dataset_slug": slug,
            "task_name": task_name,
        },
    )
    _write_asset_manifest(
        store,
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        normalized_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-normalized",
            "dataset_slug": slug,
            "task_name": task_name,
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=RAW_DATASET_ARTIFACT_TYPE,
                artifact_id=raw_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        CHUNKED_CORPUS_ARTIFACT_TYPE,
        chunks_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-chunks",
            "dataset_slug": slug,
            "task_name": task_name,
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
                artifact_id=normalized_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        EMBEDDINGS_ARTIFACT_TYPE,
        embeddings_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-embeddings",
            "dataset_slug": slug,
            "task_name": task_name,
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        es_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-es",
            "dataset_slug": slug,
            "task_name": task_name,
            "index_name": f"{slug}_{run_id}_es",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            )
        ],
    )
    _write_asset_manifest(
        store,
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        milvus_id,
        {
            "asset_fingerprint_sha256": f"fp-{slug}-milvus",
            "dataset_slug": slug,
            "task_name": task_name,
            "collection_name": f"{slug}_{run_id}_milvus",
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=chunks_id,
            ),
            ArtifactDependency(
                artifact_type=EMBEDDINGS_ARTIFACT_TYPE,
                artifact_id=embeddings_id,
            ),
        ],
    )


def _real_five_dataset_rows() -> list[tuple[str, str]]:
    return [
        ("ifir_nfcorpus", "IFIRNFCorpus"),
        ("nfcorpus", "NFCorpus"),
        ("ifir_scifact", "IFIRScifact"),
        ("scifact", "SciFact"),
        ("litsearch", "LitSearchRetrieval"),
    ]


def _real_five_dataset_asset_specs() -> list[BenchmarkDatasetSpec]:
    run_id = "e1_e4_bge_m3_fp_20260530"
    return [
        BenchmarkDatasetSpec(
            dataset_key=slug,
            task_name=task_name,
            normalized_dataset_artifact_id=f"{slug}_{run_id}_normalized",
            elasticsearch_index_artifact_id=f"{slug}_{run_id}_es_index",
            milvus_collection_artifact_id=f"{slug}_{run_id}_milvus_collection",
            index_name=f"{slug}_{run_id}_es",
            collection_name=f"{slug}_{run_id}_milvus",
        )
        for slug, task_name in _real_five_dataset_rows()
    ]
