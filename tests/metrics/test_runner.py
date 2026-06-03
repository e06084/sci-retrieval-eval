"""Tests for metrics run orchestration."""

from pathlib import Path
from typing import Any

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking.progress import ProgressEvent
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.metrics import (
    METRICS_RUN_ARTIFACT_TYPE,
    MetricsRunConfig,
    read_metrics_run_artifact,
    run_metrics,
)
from eval_platform.metrics import runner as metrics_runner
from eval_platform.retrieval import (
    RetrievalHit,
    RetrievalQueryResult,
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
            corpus=[
                CorpusRecord(doc_id="doc-1", text="one"),
                CorpusRecord(doc_id="doc-2", text="two"),
                CorpusRecord(doc_id="doc-3", text="three"),
            ],
            queries=[
                QueryRecord(query_id="q-1", text="query one"),
                QueryRecord(query_id="q-2", text="query two"),
                QueryRecord(query_id="q-3", text="query three"),
                QueryRecord(query_id="q-4", text="query four"),
            ],
            qrels=[
                QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0),
                QrelRecord(query_id="q-2", doc_id="doc-2", relevance=1.0),
                QrelRecord(query_id="q-3", doc_id="doc-3", relevance=1.0),
                QrelRecord(query_id="q-4", doc_id="doc-x", relevance=0.0),
            ],
        ),
    )
    write_retrieval_run_artifact(
        artifact_store,
        "retrieval-1",
        [
            RetrievalQueryResult(
                query_id="q-1",
                query_text="query one",
                hits=[
                    RetrievalHit(rank=1, chunk_id="c-1", doc_id="doc-1", score=10.0),
                    RetrievalHit(rank=2, chunk_id="c-2", doc_id="doc-1", score=9.0),
                    RetrievalHit(rank=3, chunk_id="c-3", doc_id="", score=8.0),
                ],
            ),
            RetrievalQueryResult(
                query_id="q-2",
                query_text="query two",
                hits=[RetrievalHit(rank=1, chunk_id="c-4", doc_id="doc-2", score=7.0)],
                error="retrieval failed",
            ),
            RetrievalQueryResult(
                query_id="q-ignored",
                query_text="ignored",
                hits=[RetrievalHit(rank=1, chunk_id="c-5", doc_id="doc-ignored", score=1.0)],
            ),
        ],
    )
    return artifact_store


def _config(**overrides: object) -> MetricsRunConfig:
    payload: dict[str, Any] = {
        "source_normalized_dataset_artifact_id": "normalized-1",
        "source_retrieval_run_artifact_id": "retrieval-1",
        "output_artifact_id": "metrics-1",
        "k_values": [1, 3],
    }
    payload.update(overrides)
    return MetricsRunConfig(**payload)


def test_metrics_run_config_sorts_and_dedupes_k_values() -> None:
    config = _config(k_values=[10, 1, 1, 3])

    assert config.k_values == [1, 3, 10]


def test_metrics_run_config_rejects_bad_k_values() -> None:
    with pytest.raises(ValueError, match="k_values"):
        _config(k_values=[])
    with pytest.raises(ValueError, match="positive"):
        _config(k_values=[0])


def test_run_metrics_computes_from_normalized_qrels_and_retrieval(
    store: LocalArtifactStore,
) -> None:
    manifest = run_metrics(store, store, _config())
    data = read_metrics_run_artifact(store, "metrics-1")

    assert store.is_complete(METRICS_RUN_ARTIFACT_TYPE, "metrics-1") is True
    assert data.aggregate["recall_at_1"] == pytest.approx(1.0 / 3.0)
    assert data.aggregate["hit_rate_at_1"] == pytest.approx(1.0 / 3.0)
    assert data.query_metrics[0].query_id == "q-1"
    assert data.query_metrics[0].projection_stats.duplicate_doc_hit_count == 1
    assert data.query_metrics[0].projection_stats.missing_doc_id_hit_count == 1
    assert data.query_metrics[1].query_id == "q-2"
    assert data.query_metrics[1].retrieval_error == "retrieval failed"
    assert data.query_metrics[2].query_id == "q-3"
    assert data.query_metrics[2].ranked_docs == []
    assert manifest.metadata["missing_result_query_count"] == 1
    assert manifest.metadata["failed_retrieval_query_count"] == 1
    assert manifest.metadata["ignored_result_query_count"] == 1
    assert manifest.metadata["skipped_no_positive_qrels_query_count"] == 1
    assert manifest.metadata["missing_doc_id_hit_count"] == 1
    assert manifest.metadata["duplicate_doc_hit_count"] == 1
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("retrieval_run", "retrieval-1"),
    ]


def test_run_metrics_reads_retrieval_artifact_without_trace(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include_trace_values: list[bool | None] = []

    def read_without_trace_probe(*args: Any, include_trace: bool = True, **kwargs: Any) -> Any:
        include_trace_values.append(include_trace)
        return read_retrieval_run_artifact(*args, include_trace=include_trace, **kwargs)

    monkeypatch.setattr(
        metrics_runner,
        "read_retrieval_run_artifact",
        read_without_trace_probe,
    )

    run_metrics(store, store, _config())

    assert include_trace_values == [False]


def test_run_metrics_reports_query_progress(store: LocalArtifactStore) -> None:
    events: list[ProgressEvent] = []

    run_metrics(store, store, _config(), progress_reporter=events.append)

    metrics_events = [event for event in events if event.stage == "metrics_run"]

    assert [(event.current, event.total) for event in metrics_events] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert [event.metadata.get("query_id") for event in metrics_events[1:]] == [
        "q-1",
        "q-2",
        "q-3",
    ]
    assert metrics_events[2].metadata["failed_retrieval_query_count"] == 1
    assert metrics_events[3].metadata["missing_result_query_count"] == 1


def test_run_metrics_is_deterministic(store: LocalArtifactStore) -> None:
    run_metrics(store, store, _config(output_artifact_id="metrics-a"))
    run_metrics(store, store, _config(output_artifact_id="metrics-b"))

    first = read_metrics_run_artifact(store, "metrics-a")
    second = read_metrics_run_artifact(store, "metrics-b")

    assert first.aggregate == second.aggregate
    assert [record.model_dump(mode="json") for record in first.query_metrics] == [
        record.model_dump(mode="json") for record in second.query_metrics
    ]
