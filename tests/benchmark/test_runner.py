"""Tests for benchmark run orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.benchmark import (
    BENCHMARK_RUN_ARTIFACT_TYPE,
    BenchmarkRunConfig,
    read_benchmark_run_artifact,
    run_benchmark,
)
from eval_platform.chunking.progress import ProgressEvent
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.metrics import MetricsRunConfig, read_metrics_run_artifact
from eval_platform.retrieval import (
    RetrievalHit,
    RetrievalQueryResult,
    RetrievalRunConfig,
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    artifact_store = LocalArtifactStore(tmp_path)
    write_normalized_dataset_artifact(
        artifact_store,
        "normalized-1",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="doc text")],
            queries=[QueryRecord(query_id="q-1", text="alpha")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
        ),
    )
    return artifact_store


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.enrich_calls: list[list[str]] = []

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        self.search_calls.append((index_name, query, top_k))
        return [
            RetrievalHit(
                rank=None,
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="chunk text",
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
        return [
            hit.model_copy(update={"doc_id": hit.doc_id or "doc-1", "text": hit.text or "text"})
            for hit in hits
        ]


class ExplodingClient:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected client call: {name}")


def _retrieval_config(**overrides: Any) -> RetrievalRunConfig:
    payload: dict[str, Any] = {
        "source_normalized_dataset_artifact_id": "normalized-1",
        "output_artifact_id": "retrieval-1",
        "retrieval_mode": "es",
        "top_k": 1,
        "elasticsearch_index_artifact_id": "es-artifact",
        "index_name": "chunks-index",
    }
    payload.update(overrides)
    if payload.get("execution_mode") == "replay" and "trace_mode" not in overrides:
        payload["trace_mode"] = "replay"
    return RetrievalRunConfig(**payload)


def _metrics_config(**overrides: Any) -> MetricsRunConfig:
    payload: dict[str, Any] = {
        "source_normalized_dataset_artifact_id": "normalized-1",
        "source_retrieval_run_artifact_id": "retrieval-1",
        "output_artifact_id": "metrics-1",
        "k_values": [1, 10],
    }
    payload.update(overrides)
    return MetricsRunConfig(**payload)


def _benchmark_config(**overrides: Any) -> BenchmarkRunConfig:
    payload: dict[str, Any] = {
        "output_artifact_id": "bench-1",
        "source_normalized_dataset_artifact_id": "normalized-1",
        "retrieval": _retrieval_config(),
        "metrics": _metrics_config(),
        "setting_name": "es-baseline",
        "description": "baseline",
        "tags": ["smoke", " smoke ", "", "baseline"],
    }
    payload.update(overrides)
    return BenchmarkRunConfig(**payload)


def test_benchmark_run_config_validates_ids_and_nested_consistency() -> None:
    with pytest.raises(ValueError, match="output_artifact_id"):
        _benchmark_config(output_artifact_id="")
    with pytest.raises(ValueError, match="retrieval.source_normalized_dataset_artifact_id"):
        _benchmark_config(retrieval=_retrieval_config(source_normalized_dataset_artifact_id="other"))
    with pytest.raises(ValueError, match="metrics.source_normalized_dataset_artifact_id"):
        _benchmark_config(metrics=_metrics_config(source_normalized_dataset_artifact_id="other"))
    with pytest.raises(ValueError, match="metrics.source_retrieval_run_artifact_id"):
        _benchmark_config(metrics=_metrics_config(source_retrieval_run_artifact_id="other"))


def test_benchmark_run_config_dedupes_tags() -> None:
    assert _benchmark_config().tags == ["smoke", "baseline"]


def test_run_benchmark_live_path_writes_summary_and_dependencies(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()

    manifest = run_benchmark(store, store, _benchmark_config(), es_client=es)
    summary = read_benchmark_run_artifact(store, "bench-1")
    retrieval_records = read_retrieval_run_artifact(store, "retrieval-1")
    metrics_data = read_metrics_run_artifact(store, "metrics-1")

    assert es.search_calls == [("chunks-index", "alpha", 1)]
    assert retrieval_records[0].hits[0].doc_id == "doc-1"
    assert metrics_data.main_score == 1.0
    assert metrics_data.main_score_metric == "recall_at_10"
    assert summary.main_score == 1.0
    assert summary.main_score_metric == "recall_at_10"
    assert summary.setting_name == "es-baseline"
    assert summary.aggregate_metrics["recall_at_10"] == 1.0
    assert summary.aggregate_metrics["ndcg_at_10"] == 1.0
    assert manifest.metadata["stage"] == "benchmark_run"
    assert manifest.metadata["setting_name"] == "es-baseline"
    assert manifest.metadata["main_score"] == 1.0
    assert manifest.metadata["retrieval_failed_query_count"] == 0
    assert manifest.metadata["metrics_evaluated_query_count"] == 1
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("retrieval_run", "retrieval-1"),
        ("metrics_run", "metrics-1"),
    ]


def test_run_benchmark_reports_stage_and_child_progress(
    store: LocalArtifactStore,
) -> None:
    events: list[ProgressEvent] = []

    run_benchmark(
        store,
        store,
        _benchmark_config(),
        es_client=FakeElasticsearchClient(),
        progress_reporter=events.append,
    )

    benchmark_events = [event for event in events if event.stage == "benchmark_run"]

    assert [(event.current, event.total) for event in benchmark_events] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert "retrieval_run" in {event.stage for event in events}
    assert "metrics_run" in {event.stage for event in events}
    assert benchmark_events[-1].metadata["main_score"] == 1.0


def test_run_benchmark_replay_path_does_not_call_external_clients(
    store: LocalArtifactStore,
) -> None:
    write_retrieval_run_artifact(
        store,
        "source-run",
        [
            RetrievalQueryResult(
                query_id="q-1",
                query_text="alpha",
                hits=[
                    RetrievalHit(
                        rank=1,
                        chunk_id="chunk-1",
                        doc_id="doc-1",
                        text="chunk text",
                        score=10.0,
                        recall_source="es",
                    )
                ],
                trace={"rewrite_queries": ["alpha"], "per_query": [], "final_hits": []},
            )
        ],
    )
    config = _benchmark_config(
        retrieval=_retrieval_config(
            execution_mode="replay",
            replay_source_retrieval_run_artifact_id="source-run",
            output_artifact_id="retrieval-replay",
        ),
        metrics=_metrics_config(source_retrieval_run_artifact_id="retrieval-replay"),
    )

    run_benchmark(
        store,
        store,
        config,
        es_client=ExplodingClient(),  # type: ignore[arg-type]
        milvus_client=ExplodingClient(),  # type: ignore[arg-type]
        embedding_client=ExplodingClient(),  # type: ignore[arg-type]
        rewrite_client=ExplodingClient(),  # type: ignore[arg-type]
        rerank_client=ExplodingClient(),  # type: ignore[arg-type]
    )

    summary = read_benchmark_run_artifact(store, "bench-1")
    assert summary.retrieval_run_artifact_id == "retrieval-replay"
    assert summary.main_score == 1.0


def test_run_benchmark_retrieval_failure_does_not_write_success(
    store: LocalArtifactStore,
) -> None:
    with pytest.raises(Exception, match="es_client"):
        run_benchmark(store, store, _benchmark_config())

    assert store.is_complete(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1") is False


def test_run_benchmark_metrics_failure_does_not_write_success(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_metrics(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("metrics failed")

    monkeypatch.setattr("eval_platform.benchmark.runner.run_metrics", fail_metrics)

    with pytest.raises(RuntimeError, match="metrics failed"):
        run_benchmark(store, store, _benchmark_config(), es_client=FakeElasticsearchClient())

    assert store.is_complete(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1") is False
