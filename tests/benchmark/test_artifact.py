"""Tests for benchmark_run artifact IO."""

import json
from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, ArtifactIncompleteError, LocalArtifactStore
from eval_platform.benchmark import (
    BENCHMARK_RUN_ARTIFACT_TYPE,
    SUMMARY_FILENAME,
    BenchmarkRunSummary,
    read_benchmark_run_artifact,
    write_benchmark_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _summary() -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        benchmark_run_artifact_id="bench-1",
        setting_name="hybrid",
        retrieval_run_artifact_id="retrieval-1",
        metrics_run_artifact_id="metrics-1",
        source_normalized_dataset_artifact_id="normalized-1",
        main_score=0.5,
        main_score_metric="ndcg_at_10",
        aggregate_metrics={"ndcg_at_10": 0.5},
    )


def test_write_benchmark_run_artifact_writes_summary_and_manifest(
    store: LocalArtifactStore,
) -> None:
    manifest = write_benchmark_run_artifact(
        store,
        "bench-1",
        _summary(),
        metadata={"setting_name": "hybrid"},
        dependencies=[
            ArtifactDependency(artifact_type="normalized_dataset", artifact_id="normalized-1"),
            ArtifactDependency(artifact_type="retrieval_run", artifact_id="retrieval-1"),
            ArtifactDependency(artifact_type="metrics_run", artifact_id="metrics-1"),
        ],
    )

    assert store.is_complete(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1") is True
    assert store.exists(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1", SUMMARY_FILENAME)
    assert manifest.metadata["stage"] == "benchmark_run"
    assert manifest.metadata["main_score"] == 0.5
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("retrieval_run", "retrieval-1"),
        ("metrics_run", "metrics-1"),
    ]


def test_read_benchmark_run_artifact_round_trips_summary(store: LocalArtifactStore) -> None:
    write_benchmark_run_artifact(store, "bench-1", _summary())

    loaded = read_benchmark_run_artifact(store, "bench-1")

    assert loaded == _summary()


def test_summary_json_does_not_include_per_query_metrics(store: LocalArtifactStore) -> None:
    write_benchmark_run_artifact(store, "bench-1", _summary())

    payload = json.loads(
        store.get_file(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1", SUMMARY_FILENAME).decode("utf-8")
    )

    assert "query_metrics" not in payload
    assert "retrieval_hits" not in payload


def test_read_benchmark_run_artifact_requires_success(store: LocalArtifactStore) -> None:
    store.put_file(BENCHMARK_RUN_ARTIFACT_TYPE, "bench-1", SUMMARY_FILENAME, b"{}")

    with pytest.raises(ArtifactIncompleteError):
        read_benchmark_run_artifact(store, "bench-1")
