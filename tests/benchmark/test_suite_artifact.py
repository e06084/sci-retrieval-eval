"""Tests for benchmark suite artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, LocalArtifactStore
from eval_platform.artifacts.types import BENCHMARK_RUN_ARTIFACT_TYPE
from eval_platform.benchmark import (
    BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
    SUITE_SUMMARY_FILENAME,
    BenchmarkSuiteArtifactError,
    BenchmarkSuiteItemSummary,
    BenchmarkSuiteRunSummary,
    read_benchmark_suite_run_artifact,
    write_benchmark_suite_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _summary() -> BenchmarkSuiteRunSummary:
    return BenchmarkSuiteRunSummary(
        suite_run_id="suite-1",
        item_count=1,
        dataset_count=1,
        setting_count=1,
        items=[
            BenchmarkSuiteItemSummary(
                dataset_key="ds1",
                setting_key="E2-es",
                benchmark_run_artifact_id="suite-1__ds1__E2-es__benchmark",
                retrieval_run_artifact_id="suite-1__ds1__E2-es__retrieval",
                metrics_run_artifact_id="suite-1__ds1__E2-es__metrics",
                main_score=1.0,
                main_score_metric="ndcg_at_10",
                aggregate_metrics={"ndcg_at_10": 1.0},
            )
        ],
    )


def test_write_and_read_benchmark_suite_run_artifact(
    store: LocalArtifactStore,
) -> None:
    manifest = write_benchmark_suite_run_artifact(
        store,
        "suite-1",
        _summary(),
        metadata={
            "datasets": [{"dataset_key": "ds1"}],
            "settings": [{"setting_key": "E2-es"}],
        },
        dependencies=[
            ArtifactDependency(
                artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
                artifact_id="suite-1__ds1__E2-es__benchmark",
            )
        ],
    )

    loaded = read_benchmark_suite_run_artifact(store, "suite-1")
    payload = json.loads(
        store.get_file(
            BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
            "suite-1",
            SUITE_SUMMARY_FILENAME,
        )
    )

    assert loaded == _summary()
    assert store.is_complete(BENCHMARK_SUITE_RUN_ARTIFACT_TYPE, "suite-1") is True
    assert manifest.metadata["stage"] == "benchmark_suite_run"
    assert manifest.metadata["suite_run_id"] == "suite-1"
    assert manifest.metadata["dataset_count"] == 1
    assert manifest.metadata["setting_count"] == 1
    assert manifest.metadata["item_count"] == 1
    assert manifest.metadata["datasets"] == [{"dataset_key": "ds1"}]
    assert manifest.metadata["settings"] == [{"setting_key": "E2-es"}]
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("benchmark_run", "suite-1__ds1__E2-es__benchmark")
    ]
    assert "query_metrics" not in payload
    assert "hits" not in payload


def test_write_benchmark_suite_run_artifact_validates_summary_id(
    store: LocalArtifactStore,
) -> None:
    with pytest.raises(BenchmarkSuiteArtifactError, match="suite_run_id"):
        write_benchmark_suite_run_artifact(store, "other-suite", _summary())
