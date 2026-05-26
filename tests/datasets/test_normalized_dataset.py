"""Tests for normalized dataset artifact helpers."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactIncompleteError,
    ArtifactManifest,
    LocalArtifactStore,
)
from eval_platform.datasets import (
    CORPUS_FILENAME,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    QRELS_FILENAME,
    QUERIES_FILENAME,
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    dump_jsonl,
    read_normalized_dataset_artifact,
    write_normalized_dataset_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _sample_dataset() -> NormalizedDataset:
    return NormalizedDataset(
        corpus=[
            CorpusRecord(doc_id="doc-1", text="corpus text", title="Title"),
            CorpusRecord(doc_id="doc-2", text="another doc"),
        ],
        queries=[QueryRecord(query_id="q-1", text="query text")],
        qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=2.0)],
        metadata={"source": "unit-test"},
    )


def test_write_normalized_dataset_artifact_writes_files(store: LocalArtifactStore) -> None:
    dataset = _sample_dataset()

    write_normalized_dataset_artifact(store, "sample_001", dataset)

    assert store.exists(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001", CORPUS_FILENAME)
    assert store.exists(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001", QUERIES_FILENAME)
    assert store.exists(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001", QRELS_FILENAME)


def test_write_normalized_dataset_artifact_marks_complete(store: LocalArtifactStore) -> None:
    write_normalized_dataset_artifact(store, "sample_001", _sample_dataset())

    assert store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001") is True


def test_read_normalized_dataset_artifact_round_trip(store: LocalArtifactStore) -> None:
    dataset = _sample_dataset()

    write_normalized_dataset_artifact(store, "sample_001", dataset)
    loaded = read_normalized_dataset_artifact(store, "sample_001")

    assert loaded == dataset


def test_manifest_metadata_contains_counts(store: LocalArtifactStore) -> None:
    dataset = _sample_dataset()

    manifest = write_normalized_dataset_artifact(
        store,
        "sample_001",
        dataset,
        metadata={"source": "unit-test"},
    )

    assert manifest.metadata["corpus_count"] == 2
    assert manifest.metadata["query_count"] == 1
    assert manifest.metadata["qrel_count"] == 1
    assert manifest.metadata["source"] == "unit-test"


def test_manifest_count_metadata_is_not_overridden_by_user_metadata(
    store: LocalArtifactStore,
) -> None:
    dataset = NormalizedDataset(
        corpus=[CorpusRecord(doc_id="doc-1", text="corpus text")],
        queries=[QueryRecord(query_id="q-1", text="query text")],
        qrels=[QrelRecord(query_id="q-1", doc_id="doc-1")],
        metadata={
            "corpus_count": 999,
            "query_count": 999,
            "qrel_count": 999,
        },
    )

    manifest = write_normalized_dataset_artifact(
        store,
        "sample_001",
        dataset,
        metadata={
            "corpus_count": 888,
            "query_count": 888,
            "qrel_count": 888,
        },
    )

    assert manifest.metadata["corpus_count"] == 1
    assert manifest.metadata["query_count"] == 1
    assert manifest.metadata["qrel_count"] == 1


def test_manifest_files_include_dataset_jsonl_files(store: LocalArtifactStore) -> None:
    manifest = write_normalized_dataset_artifact(store, "sample_001", _sample_dataset())

    file_paths = {file.path for file in manifest.files}

    assert file_paths == {CORPUS_FILENAME, QUERIES_FILENAME, QRELS_FILENAME}


def test_read_requires_complete_artifact(store: LocalArtifactStore) -> None:
    dataset = _sample_dataset()
    artifact_id = "sample_001"

    store.put_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        artifact_id,
        CORPUS_FILENAME,
        dump_jsonl(dataset.corpus).encode("utf-8"),
    )
    store.put_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        artifact_id,
        QUERIES_FILENAME,
        dump_jsonl(dataset.queries).encode("utf-8"),
    )
    store.put_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        artifact_id,
        QRELS_FILENAME,
        dump_jsonl(dataset.qrels).encode("utf-8"),
    )
    store.write_manifest(
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        artifact_id,
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
            created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
        ),
    )

    with pytest.raises(ArtifactIncompleteError):
        read_normalized_dataset_artifact(store, artifact_id)


def test_success_marker_written_after_manifest(store: LocalArtifactStore) -> None:
    write_normalized_dataset_artifact(store, "sample_001", _sample_dataset())

    manifest = store.read_manifest(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001")

    assert manifest.artifact_id == "sample_001"
    assert store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "sample_001") is True


def test_write_normalized_dataset_artifact_accepts_dependencies(
    store: LocalArtifactStore,
) -> None:
    manifest = write_normalized_dataset_artifact(
        store,
        "sample_001",
        _sample_dataset(),
        dependencies=[ArtifactDependency(artifact_type="raw_dataset", artifact_id="raw_001")],
    )

    assert manifest.dependencies == [
        ArtifactDependency(artifact_type="raw_dataset", artifact_id="raw_001")
    ]
