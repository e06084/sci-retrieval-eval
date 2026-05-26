"""Tests for retrieval_run artifact IO."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, ArtifactIncompleteError, LocalArtifactStore
from eval_platform.retrieval import (
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    RetrievalHit,
    RetrievalQueryResult,
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _records() -> list[RetrievalQueryResult]:
    return [
        RetrievalQueryResult(
            query_id="q-1",
            query_text="query one",
            hits=[
                RetrievalHit(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="chunk text",
                    score=1.0,
                    recall_source="es",
                )
            ],
        ),
        RetrievalQueryResult(query_id="q-2", query_text="query two", hits=[], error="failed"),
        RetrievalQueryResult(query_id="q-3", query_text="query three", hits=[]),
    ]


def test_write_retrieval_run_artifact_writes_sharded_results(
    store: LocalArtifactStore,
) -> None:
    manifest = write_retrieval_run_artifact(
        store,
        "run-1",
        _records(),
        queries_per_shard=2,
        metadata={"retrieval_mode": "hybrid"},
        dependencies=[
            ArtifactDependency(
                artifact_type="normalized_dataset",
                artifact_id="normalized-1",
            )
        ],
        created_at=datetime(2026, 5, 27, tzinfo=UTC),
    )

    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "run-1") is True
    assert [file.path for file in manifest.files] == [
        "results/part-00000.jsonl",
        "results/part-00001.jsonl",
    ]
    assert manifest.metadata["query_count"] == 3
    assert manifest.metadata["succeeded_query_count"] == 2
    assert manifest.metadata["failed_query_count"] == 1
    assert manifest.metadata["result_file_count"] == 2
    assert manifest.metadata["result_record_count"] == 3
    assert manifest.dependencies[0].artifact_id == "normalized-1"


def test_read_retrieval_run_artifact_round_trips_records(store: LocalArtifactStore) -> None:
    records = _records()
    write_retrieval_run_artifact(store, "run-1", records, queries_per_shard=2)

    loaded = read_retrieval_run_artifact(store, "run-1")

    assert [record.query_id for record in loaded] == ["q-1", "q-2", "q-3"]
    assert loaded[0].hits[0].chunk_id == "chunk-1"
    assert loaded[1].error == "failed"


def test_read_retrieval_run_artifact_requires_success(store: LocalArtifactStore) -> None:
    store.put_file(RETRIEVAL_RUN_ARTIFACT_TYPE, "run-1", "results/part-00000.jsonl", b"")

    with pytest.raises(ArtifactIncompleteError):
        read_retrieval_run_artifact(store, "run-1")
