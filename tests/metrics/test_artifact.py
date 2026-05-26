"""Tests for metrics_run artifact IO."""

from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, ArtifactIncompleteError, LocalArtifactStore
from eval_platform.metrics import (
    METRICS_FILENAME,
    METRICS_RUN_ARTIFACT_TYPE,
    QUERY_METRICS_DIR,
    MetricsRunData,
    ProjectionStats,
    QueryMetricsRecord,
    RankedDoc,
    read_metrics_run_artifact,
    write_metrics_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _query_record(query_id: str) -> QueryMetricsRecord:
    return QueryMetricsRecord(
        query_id=query_id,
        query_text="query",
        ranked_docs=[
            RankedDoc(
                rank=1,
                doc_id="doc-1",
                score=1.0,
                source_chunk_id="chunk-1",
                source_chunk_rank=1,
                source_chunk_score=9.0,
            )
        ],
        relevant_docs={"doc-1": 1.0},
        metrics={"ndcg_at_1": 1.0},
        projection_stats=ProjectionStats(
            input_hit_count=1,
            ranked_doc_count=1,
            missing_doc_id_hit_count=0,
            duplicate_doc_hit_count=0,
        ),
    )


def test_write_metrics_run_artifact_writes_summary_and_shards(
    store: LocalArtifactStore,
) -> None:
    manifest = write_metrics_run_artifact(
        store,
        "metrics-1",
        MetricsRunData(
            aggregate={"ndcg_at_1": 1.0},
            k_values=[1],
            main_score=1.0,
            main_score_metric="ndcg_at_1",
            query_metrics=[_query_record("q-1"), _query_record("q-2")],
            metadata={"source_retrieval_run_artifact_id": "retrieval-1"},
        ),
        queries_per_shard=1,
        metadata={"queries_per_shard": 999},
        dependencies=[
            ArtifactDependency(artifact_type="normalized_dataset", artifact_id="normalized-1"),
            ArtifactDependency(artifact_type="retrieval_run", artifact_id="retrieval-1"),
        ],
    )

    assert store.is_complete(METRICS_RUN_ARTIFACT_TYPE, "metrics-1") is True
    assert store.exists(METRICS_RUN_ARTIFACT_TYPE, "metrics-1", METRICS_FILENAME)
    assert store.exists(
        METRICS_RUN_ARTIFACT_TYPE,
        "metrics-1",
        f"{QUERY_METRICS_DIR}/part-00000.jsonl",
    )
    assert manifest.metadata["stage"] == "metrics_run"
    assert manifest.metadata["queries_per_shard"] == 1
    assert manifest.metadata["query_metric_file_count"] == 2
    assert manifest.metadata["query_metric_record_count"] == 2
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("retrieval_run", "retrieval-1"),
    ]


def test_read_metrics_run_artifact_round_trips(store: LocalArtifactStore) -> None:
    data = MetricsRunData(
        aggregate={"ndcg_at_1": 1.0},
        k_values=[1],
        main_score=1.0,
        main_score_metric="ndcg_at_1",
        query_metrics=[_query_record("q-1")],
        metadata={"stage_note": "test"},
    )
    write_metrics_run_artifact(store, "metrics-1", data)

    loaded = read_metrics_run_artifact(store, "metrics-1")

    assert loaded.aggregate == {"ndcg_at_1": 1.0}
    assert loaded.query_metrics[0].query_id == "q-1"
    assert loaded.metadata["stage_note"] == "test"


def test_read_metrics_run_artifact_requires_success(store: LocalArtifactStore) -> None:
    store.put_file(METRICS_RUN_ARTIFACT_TYPE, "metrics-1", METRICS_FILENAME, b"{}")

    with pytest.raises(ArtifactIncompleteError):
        read_metrics_run_artifact(store, "metrics-1")
