"""Tests for LocalArtifactStore."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactNotFoundError,
    InvalidArtifactPathError,
    LocalArtifactStore,
    ManifestMismatchError,
    ManifestNotFoundError,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _sample_manifest(artifact_id: str = "sample_001") -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type="normalized_dataset",
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
        created_by="test",
        files=[ArtifactFile(path="corpus.jsonl", size_bytes=128, sha256="abc123")],
    )


def test_write_and_read_manifest(store: LocalArtifactStore) -> None:
    manifest = _sample_manifest()

    store.write_manifest("normalized_dataset", "sample_001", manifest)
    loaded = store.read_manifest("normalized_dataset", "sample_001")

    assert loaded == manifest


def test_is_complete_false_with_manifest_only(store: LocalArtifactStore) -> None:
    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest())

    assert store.is_complete("normalized_dataset", "sample_001") is False


def test_is_complete_false_with_success_only(store: LocalArtifactStore) -> None:
    store.mark_success("normalized_dataset", "sample_001")

    assert store.is_complete("normalized_dataset", "sample_001") is False


def test_is_complete_true_with_manifest_and_success(store: LocalArtifactStore) -> None:
    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest())
    store.mark_success("normalized_dataset", "sample_001")

    assert store.is_complete("normalized_dataset", "sample_001") is True


def test_read_manifest_raises_when_missing(store: LocalArtifactStore) -> None:
    with pytest.raises(ManifestNotFoundError):
        store.read_manifest("normalized_dataset", "missing")


def test_put_get_and_exists(store: LocalArtifactStore) -> None:
    store.put_file("normalized_dataset", "sample_001", "corpus.jsonl", b'{"id": 1}\n')

    assert store.exists("normalized_dataset", "sample_001", "corpus.jsonl") is True
    assert store.get_file("normalized_dataset", "sample_001", "corpus.jsonl") == b'{"id": 1}\n'


def test_get_file_raises_when_missing(store: LocalArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        store.get_file("normalized_dataset", "sample_001", "missing.jsonl")


def test_list_artifacts(store: LocalArtifactStore) -> None:
    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest("sample_001"))
    store.write_manifest("normalized_dataset", "sample_002", _sample_manifest("sample_002"))
    store.write_manifest(
        "chunked_corpus",
        "sample_003",
        ArtifactManifest(
            artifact_id="sample_003",
            artifact_type="chunked_corpus",
            created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
        ),
    )

    assert store.list_artifacts() == [
        ("chunked_corpus", "sample_003"),
        ("normalized_dataset", "sample_001"),
        ("normalized_dataset", "sample_002"),
    ]
    assert store.list_artifacts("normalized_dataset") == [
        ("normalized_dataset", "sample_001"),
        ("normalized_dataset", "sample_002"),
    ]


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.txt", "/tmp/evil.txt", "nested/../../outside.txt"],
)
def test_rejects_unsafe_relative_path(store: LocalArtifactStore, relative_path: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file("normalized_dataset", "sample_001", relative_path, b"data")


@pytest.mark.parametrize(
    "artifact_type",
    ["../escape", "../../x", "bad/type"],
)
def test_rejects_unsafe_artifact_type(store: LocalArtifactStore, artifact_type: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file(artifact_type, "sample_001", "corpus.jsonl", b"data")


@pytest.mark.parametrize(
    "artifact_id",
    ["../escape", "../../y", "bad/id"],
)
def test_rejects_unsafe_artifact_id(store: LocalArtifactStore, artifact_id: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file("normalized_dataset", artifact_id, "corpus.jsonl", b"data")


def test_write_manifest_rejects_mismatched_type(store: LocalArtifactStore) -> None:
    manifest = _sample_manifest()

    with pytest.raises(ManifestMismatchError):
        store.write_manifest("chunked_corpus", "sample_001", manifest)


def test_write_manifest_rejects_mismatched_id(store: LocalArtifactStore) -> None:
    manifest = _sample_manifest("sample_001")

    with pytest.raises(ManifestMismatchError):
        store.write_manifest("normalized_dataset", "other_id", manifest)


def test_artifact_uri_returns_file_uri(store: LocalArtifactStore, tmp_path: Path) -> None:
    uri = store.artifact_uri("normalized_dataset", "sample_001")

    assert uri == store.artifact_dir("normalized_dataset", "sample_001").resolve().as_uri()
    assert uri.startswith("file://")
    assert str(tmp_path.resolve()) in uri
