"""Tests for benchmark suite config and runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.artifacts.types import (
    BENCHMARK_RUN_ARTIFACT_TYPE,
    BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
)
from eval_platform.benchmark import (
    SUITE_SUMMARY_FILENAME,
    BenchmarkDatasetSpec,
    BenchmarkSettingSpec,
    BenchmarkSuiteRunConfig,
    build_benchmark_run_config,
    read_benchmark_suite_run_artifact,
    run_benchmark_suite,
    settings_for_selection,
)
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.retrieval import RetrievalHit


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    artifact_store = LocalArtifactStore(tmp_path)
    _write_dataset(artifact_store, "normalized-ds1", "q-1", "alpha", "doc-alpha")
    _write_dataset(artifact_store, "normalized-ds2", "q-2", "beta", "doc-beta")
    return artifact_store


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0 if text == "alpha" else 2.0] for text in texts]


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.enrich_calls: list[list[str]] = []

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        self.search_calls.append((index_name, query, top_k))
        return [
            RetrievalHit(
                chunk_id=f"es-{query}",
                doc_id=f"doc-{query}",
                text=f"es text {query}",
                score=10.0,
                recall_source="es",
            )
        ]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        self.enrich_calls.append([hit.chunk_id for hit in hits])
        out: list[RetrievalHit] = []
        for hit in hits:
            doc_id = hit.doc_id
            if not doc_id:
                if hit.chunk_id == "mv-1":
                    doc_id = "doc-alpha"
                elif hit.chunk_id == "mv-2":
                    doc_id = "doc-beta"
            out.append(
                hit.model_copy(
                    update={
                        "doc_id": doc_id,
                        "text": hit.text or f"text {hit.chunk_id}",
                    }
                )
            )
        return out


class FakeMilvusClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float], int]] = []

    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        self.calls.append((collection_name, list(vector), top_k))
        return [
            RetrievalHit(
                chunk_id=f"mv-{int(vector[0])}",
                doc_id="",
                score=1.0,
                recall_source="milvus",
            )
        ]


class FakeRerankClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        self.calls.append((query, [hit.chunk_id for hit in hits], top_n))
        return list(hits[:top_n])


def _write_dataset(
    store: LocalArtifactStore,
    artifact_id: str,
    query_id: str,
    query_text: str,
    relevant_doc_id: str,
) -> None:
    write_normalized_dataset_artifact(
        store,
        artifact_id,
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id=relevant_doc_id, text=f"text {relevant_doc_id}")],
            queries=[QueryRecord(query_id=query_id, text=query_text)],
            qrels=[QrelRecord(query_id=query_id, doc_id=relevant_doc_id, relevance=1.0)],
        ),
    )


def _dataset(
    dataset_key: str = "ds1",
    normalized_id: str = "normalized-ds1",
) -> BenchmarkDatasetSpec:
    return BenchmarkDatasetSpec(
        dataset_key=dataset_key,
        task_name=f"Task {dataset_key}",
        normalized_dataset_artifact_id=normalized_id,
        elasticsearch_index_artifact_id=f"{dataset_key}-es-artifact",
        milvus_collection_artifact_id=f"{dataset_key}-milvus-artifact",
        index_name=f"{dataset_key}-index",
        collection_name=f"{dataset_key}-collection",
        metadata={"kind": "test"},
    )


def _suite_config(
    *,
    suite_run_id: str = "suite-1",
    datasets: list[BenchmarkDatasetSpec] | None = None,
    settings: list[BenchmarkSettingSpec] | None = None,
    query_limit: int | None = None,
) -> BenchmarkSuiteRunConfig:
    return BenchmarkSuiteRunConfig(
        suite_run_id=suite_run_id,
        datasets=datasets or [_dataset()],
        settings=settings or settings_for_selection(),
        metrics_k_values=[1, 10],
        query_limit=query_limit,
    )


def test_suite_config_validates_keys_and_duplicates() -> None:
    with pytest.raises(ValueError, match="dataset_key"):
        _dataset(dataset_key="")
    with pytest.raises(ValueError, match="setting_key"):
        _suite_config(settings=[BenchmarkSettingSpec(setting_key="bad key", retrieval_mode="es")])
    with pytest.raises(ValueError, match="duplicate dataset_key"):
        _suite_config(datasets=[_dataset("ds1"), _dataset("ds1", "normalized-ds2")])
    with pytest.raises(ValueError, match="duplicate setting_key"):
        _suite_config(
            settings=[
                BenchmarkSettingSpec(setting_key="E2-es", retrieval_mode="es"),
                BenchmarkSettingSpec(setting_key="E2-es", retrieval_mode="es"),
            ]
        )


def test_suite_config_validates_query_limit() -> None:
    assert _suite_config(query_limit=3).query_limit == 3
    with pytest.raises(ValueError, match="query_limit"):
        _suite_config(query_limit=0)
    with pytest.raises(ValueError, match="query_limit"):
        _suite_config(query_limit=-1)


def test_build_benchmark_run_config_generates_stable_artifact_ids() -> None:
    config = _suite_config(settings=settings_for_selection("E3-hybrid"))
    item = build_benchmark_run_config(
        config,
        config.datasets[0],
        config.settings[0],
    )

    assert item.output_artifact_id == "suite-1__ds1__E3-hybrid__benchmark"
    assert item.retrieval.output_artifact_id == "suite-1__ds1__E3-hybrid__retrieval"
    assert item.metrics.output_artifact_id == "suite-1__ds1__E3-hybrid__metrics"
    assert " " not in item.output_artifact_id


def test_build_benchmark_run_config_passes_query_limit_to_retrieval_only() -> None:
    config = _suite_config(query_limit=3, settings=settings_for_selection("E2-es"))
    item = build_benchmark_run_config(config, config.datasets[0], config.settings[0])

    assert item.retrieval.query_limit == 3
    assert not hasattr(item.metrics, "query_limit")

    unlimited_config = _suite_config(settings=settings_for_selection("E2-es"))
    unlimited_item = build_benchmark_run_config(
        unlimited_config,
        unlimited_config.datasets[0],
        unlimited_config.settings[0],
    )
    assert unlimited_item.retrieval.query_limit is None


def test_build_benchmark_run_config_maps_e1_to_e4_retrieval_settings() -> None:
    config = _suite_config()
    dataset = config.datasets[0]
    configs = {
        setting.setting_key: build_benchmark_run_config(config, dataset, setting)
        for setting in config.settings
    }

    e1 = configs["E1-milvus"].retrieval
    assert e1.retrieval_mode == "milvus"
    assert e1.elasticsearch_index_artifact_id == "ds1-es-artifact"
    assert e1.milvus_collection_artifact_id == "ds1-milvus-artifact"
    assert e1.index_name == "ds1-index"
    assert e1.collection_name == "ds1-collection"
    assert e1.rewrite_enabled is False
    assert e1.rerank_enabled is False

    e2 = configs["E2-es"].retrieval
    assert e2.retrieval_mode == "es"
    assert e2.elasticsearch_index_artifact_id == "ds1-es-artifact"
    assert e2.milvus_collection_artifact_id is None
    assert e2.collection_name is None

    e3 = configs["E3-hybrid"].retrieval
    assert e3.retrieval_mode == "hybrid"
    assert e3.elasticsearch_index_artifact_id == "ds1-es-artifact"
    assert e3.milvus_collection_artifact_id == "ds1-milvus-artifact"

    e4 = configs["E4-hybrid-rerank"].retrieval
    assert e4.retrieval_mode == "hybrid"
    assert e4.rerank_enabled is True
    assert e4.rewrite_enabled is False
    assert e4.sub_queries == 0


def test_run_benchmark_suite_runs_items_and_writes_stable_summary(
    store: LocalArtifactStore,
) -> None:
    config = _suite_config(
        datasets=[
            _dataset("ds1", "normalized-ds1"),
            _dataset("ds2", "normalized-ds2"),
        ],
        settings=settings_for_selection(["E2-es", "E3-hybrid"]),
    )

    manifest = run_benchmark_suite(
        store,
        store,
        config,
        es_client=FakeElasticsearchClient(),
        milvus_client=FakeMilvusClient(),
        embedding_client=FakeEmbeddingClient(),
    )
    summary = read_benchmark_suite_run_artifact(store, "suite-1")
    raw_summary = json.loads(
        store.get_file(
            BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
            "suite-1",
            SUITE_SUMMARY_FILENAME,
        )
    )

    assert len(store.list_artifacts(BENCHMARK_RUN_ARTIFACT_TYPE)) == 4
    assert summary.item_count == 4
    assert summary.dataset_count == 2
    assert summary.setting_count == 2
    assert [(item.dataset_key, item.setting_key) for item in summary.items] == [
        ("ds1", "E2-es"),
        ("ds1", "E3-hybrid"),
        ("ds2", "E2-es"),
        ("ds2", "E3-hybrid"),
    ]
    assert all(item.main_score_metric == "ndcg_at_10" for item in summary.items)
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("benchmark_run", item.benchmark_run_artifact_id)
        for item in summary.items
    ]
    assert manifest.metadata["stage"] == "benchmark_suite_run"
    assert manifest.metadata["dataset_count"] == 2
    assert manifest.metadata["setting_count"] == 2
    assert manifest.metadata["item_count"] == 4
    assert manifest.metadata["query_limit"] is None
    assert store.is_complete(BENCHMARK_SUITE_RUN_ARTIFACT_TYPE, "suite-1") is True
    assert "query_metrics" not in raw_summary
    assert "hits" not in raw_summary


def test_run_benchmark_suite_records_query_limit_in_manifest(
    store: LocalArtifactStore,
) -> None:
    config = _suite_config(query_limit=1, settings=settings_for_selection("E2-es"))

    manifest = run_benchmark_suite(
        store,
        store,
        config,
        es_client=FakeElasticsearchClient(),
    )

    assert manifest.metadata["query_limit"] == 1


def test_run_benchmark_suite_failure_does_not_write_success_marker(
    store: LocalArtifactStore,
) -> None:
    config = _suite_config(settings=settings_for_selection("E2-es"))

    with pytest.raises(Exception, match="es_client"):
        run_benchmark_suite(store, store, config)

    assert store.is_complete(BENCHMARK_SUITE_RUN_ARTIFACT_TYPE, "suite-1") is False
