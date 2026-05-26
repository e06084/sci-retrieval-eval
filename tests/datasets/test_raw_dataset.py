"""Tests for raw dataset snapshot artifact helpers."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactIncompleteError,
    ArtifactManifest,
    LocalArtifactStore,
    S3ArtifactStore,
)
from eval_platform.datasets import (
    RAW_DATASET_ARTIFACT_TYPE,
    RawDatasetArtifactError,
    RawDatasetFile,
    RawDatasetSnapshot,
    build_content_fingerprint_sha256,
    import_raw_dataset_from_local_dir,
    import_raw_dataset_from_s3_prefix,
    read_raw_dataset_artifact,
    write_raw_dataset_artifact,
)


class RecordingStreamingBody:
    def __init__(self, payload: bytes, chunk_size: int = 2) -> None:
        self._payload = payload
        self._chunk_size = chunk_size
        self._offset = 0
        self.read_calls = 0
        self.max_requested = 0

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        self.max_requested = max(self.max_requested, size)
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        actual_size = min(size, self._chunk_size, len(self._payload) - self._offset)
        chunk = self._payload[self._offset : self._offset + actual_size]
        self._offset += actual_size
        return chunk


class FakeS3Client:
    def __init__(self, page_size: int | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size
        self.streaming_bodies: dict[str, RecordingStreamingBody] = {}
        self.streaming_keys: set[str] = set()

    def _full_key(self, bucket: str, key: str) -> str:
        return f"{bucket}/{key}"

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | io.BytesIO) -> None:
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[self._full_key(Bucket, Key)] = data

    def get_object(
        self, *, Bucket: str, Key: str
    ) -> dict[str, io.BytesIO | RecordingStreamingBody]:
        full_key = self._full_key(Bucket, Key)
        if full_key not in self.streaming_keys:
            return {"Body": io.BytesIO(self.objects[full_key])}
        body = RecordingStreamingBody(self.objects[full_key])
        self.streaming_bodies[full_key] = body
        return {"Body": body}

    def head_object(self, *, Bucket: str, Key: str) -> None:
        if self._full_key(Bucket, Key) not in self.objects:
            raise KeyError(Key)

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        matching_keys = sorted(
            key[len(f"{Bucket}/") :]
            for key in self.objects
            if key.startswith(f"{Bucket}/") and key[len(f"{Bucket}/") :].startswith(Prefix)
        )
        page_size = self.page_size or len(matching_keys)
        start = int(ContinuationToken) if ContinuationToken else 0
        page_keys = matching_keys[start : start + page_size]
        next_start = start + page_size
        is_truncated = next_start < len(matching_keys)
        return {
            "Contents": [{"Key": key} for key in page_keys],
            "IsTruncated": is_truncated,
            "NextContinuationToken": str(next_start) if is_truncated else None,
        }


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sample_snapshot() -> RawDatasetSnapshot:
    files = [
        RawDatasetFile(
            path="corpus/a.jsonl",
            uri="s3://bucket/raw/corpus/a.jsonl",
            size_bytes=len(b'{"id": 1}\n'),
            sha256=_payload_sha256(b'{"id": 1}\n'),
        ),
        RawDatasetFile(
            path="corpus/b.jsonl",
            uri="s3://bucket/raw/corpus/b.jsonl",
            size_bytes=len(b'{"id": 2}\n'),
            sha256=_payload_sha256(b'{"id": 2}\n'),
        ),
    ]
    return RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri="s3://bucket/raw",
        dataset_name="sample-dataset",
        dataset_revision="r1",
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters={"pattern": "*.jsonl"},
        metadata={"owner": "unit-test"},
    )


def test_write_raw_dataset_artifact_is_snapshot_only(store: LocalArtifactStore) -> None:
    snapshot = _sample_snapshot()

    write_raw_dataset_artifact(store, "sample_001", snapshot)

    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "_MANIFEST.json")
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "_SUCCESS")
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "files/corpus/a.jsonl") is False


def test_write_raw_dataset_artifact_marks_complete(store: LocalArtifactStore) -> None:
    write_raw_dataset_artifact(store, "sample_001", _sample_snapshot())

    assert store.is_complete(RAW_DATASET_ARTIFACT_TYPE, "sample_001") is True


def test_read_raw_dataset_artifact_round_trip_metadata(store: LocalArtifactStore) -> None:
    snapshot = _sample_snapshot()

    write_raw_dataset_artifact(store, "sample_001", snapshot)
    loaded = read_raw_dataset_artifact(store, "sample_001")

    assert loaded == snapshot


def test_manifest_metadata_contains_required_fields(store: LocalArtifactStore) -> None:
    snapshot = _sample_snapshot()

    manifest = write_raw_dataset_artifact(
        store,
        "sample_001",
        snapshot,
        metadata={"stage": "wrong", "file_count": 999, "note": "kept"},
    )

    assert manifest.metadata["stage"] == "raw_dataset"
    assert manifest.metadata["source_type"] == "s3_prefix"
    assert manifest.metadata["source_uri"] == "s3://bucket/raw"
    assert manifest.metadata["dataset_name"] == "sample-dataset"
    assert manifest.metadata["dataset_revision"] == "r1"
    assert manifest.metadata["file_count"] == 2
    assert manifest.metadata["total_size_bytes"] == sum(
        file.size_bytes for file in snapshot.files
    )
    assert manifest.metadata["content_fingerprint_sha256"] == snapshot.content_fingerprint_sha256
    assert manifest.metadata["import_parameters"] == {"pattern": "*.jsonl"}
    assert manifest.metadata["note"] == "kept"
    assert manifest.metadata["files"][0]["uri"] == "s3://bucket/raw/corpus/a.jsonl"


def test_manifest_files_is_empty_for_snapshot_only(store: LocalArtifactStore) -> None:
    manifest = write_raw_dataset_artifact(store, "sample_001", _sample_snapshot())

    assert manifest.files == []


def test_fingerprint_mismatch_writes_no_raw_files(store: LocalArtifactStore) -> None:
    snapshot = _sample_snapshot().model_copy(
        update={"content_fingerprint_sha256": "bad-fingerprint"}
    )

    with pytest.raises(RawDatasetArtifactError):
        write_raw_dataset_artifact(store, "sample_001", snapshot)

    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "_MANIFEST.json") is False
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "_SUCCESS") is False
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "sample_001", "files/corpus/a.jsonl") is False


def test_read_requires_complete_artifact(store: LocalArtifactStore) -> None:
    snapshot = _sample_snapshot()
    artifact_id = "sample_001"

    store.write_manifest(
        RAW_DATASET_ARTIFACT_TYPE,
        artifact_id,
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=RAW_DATASET_ARTIFACT_TYPE,
            created_at=datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC),
            metadata={
                "stage": "raw_dataset",
                "source_type": snapshot.source_type,
                "source_uri": snapshot.source_uri,
                "dataset_name": snapshot.dataset_name,
                "dataset_revision": snapshot.dataset_revision,
                "file_count": 2,
                "total_size_bytes": sum(file.size_bytes for file in snapshot.files),
                "files": [file.model_dump(mode="json") for file in snapshot.files],
                "content_fingerprint_sha256": snapshot.content_fingerprint_sha256,
                "import_parameters": snapshot.import_parameters,
            },
        ),
    )

    with pytest.raises(ArtifactIncompleteError):
        read_raw_dataset_artifact(store, artifact_id)


def test_import_raw_dataset_from_local_dir_builds_snapshot_only(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source"
    (source_dir / "nested").mkdir(parents=True)
    (source_dir / "a.txt").write_bytes(b"alpha")
    (source_dir / "nested" / "b.bin").write_bytes(b"beta")

    manifest = import_raw_dataset_from_local_dir(
        store,
        "raw_local_001",
        source_dir,
        dataset_name="demo-local",
        dataset_revision="2026-05-26",
        import_parameters={"source_split": "train"},
        metadata={"team": "search"},
    )
    loaded = read_raw_dataset_artifact(store, "raw_local_001")

    assert manifest.metadata["stage"] == "raw_dataset"
    assert manifest.metadata["source_type"] == "local_dir"
    assert manifest.metadata["dataset_name"] == "demo-local"
    assert manifest.metadata["file_count"] == 2
    assert manifest.metadata["total_size_bytes"] == 9
    assert [file.path for file in loaded.files] == ["a.txt", "nested/b.bin"]
    assert loaded.files[0].uri.startswith("file://")
    assert loaded.metadata == {"team": "search"}
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "raw_local_001", "files/a.txt") is False


def test_import_raw_dataset_from_s3_prefix_to_local_store_is_snapshot_only(
    store: LocalArtifactStore,
) -> None:
    client = FakeS3Client()
    client.put_object(Bucket="raw-bucket", Key="incoming/raw/a.jsonl", Body=b"a")
    client.put_object(Bucket="raw-bucket", Key="incoming/raw/nested/b.jsonl", Body=b"bb")

    manifest = import_raw_dataset_from_s3_prefix(
        store,
        "raw_s3_001",
        client=client,
        bucket="raw-bucket",
        prefix="incoming/raw",
        dataset_name="demo-s3",
        import_parameters={"compression": "none"},
    )
    loaded = read_raw_dataset_artifact(store, "raw_s3_001")

    assert manifest.metadata["source_type"] == "s3_prefix"
    assert manifest.metadata["source_uri"] == "s3://raw-bucket/incoming/raw"
    assert manifest.metadata["file_count"] == 2
    assert [file.path for file in loaded.files] == ["a.jsonl", "nested/b.jsonl"]
    assert loaded.files[1].uri == "s3://raw-bucket/incoming/raw/nested/b.jsonl"
    assert store.exists(RAW_DATASET_ARTIFACT_TYPE, "raw_s3_001", "files/nested/b.jsonl") is False


def test_import_raw_dataset_from_s3_prefix_streams_without_collecting_full_bytes(
    store: LocalArtifactStore,
) -> None:
    client = FakeS3Client()
    client.put_object(Bucket="raw-bucket", Key="incoming/raw/large.bin", Body=b"abcdef")
    client.streaming_keys.add("raw-bucket/incoming/raw/large.bin")

    import_raw_dataset_from_s3_prefix(
        store,
        "raw_s3_streaming_001",
        client=client,
        bucket="raw-bucket",
        prefix="incoming/raw",
        dataset_name="demo-s3",
    )

    body = client.streaming_bodies["raw-bucket/incoming/raw/large.bin"]
    assert body.read_calls >= 3
    assert body.max_requested > 0
    assert (
        store.exists(RAW_DATASET_ARTIFACT_TYPE, "raw_s3_streaming_001", "files/large.bin")
        is False
    )


def test_fingerprint_is_sensitive_to_uri_change() -> None:
    base = _sample_snapshot()
    changed_uri_files = [
        file if file.path != "corpus/a.jsonl"
        else file.model_copy(update={"uri": "s3://other/raw/corpus/a.jsonl"})
        for file in base.files
    ]

    assert build_content_fingerprint_sha256(base.files) != build_content_fingerprint_sha256(
        changed_uri_files
    )


def test_import_raw_dataset_can_write_manifest_to_s3_store(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.txt").write_bytes(b"payload")

    client = FakeS3Client()
    store = S3ArtifactStore(bucket="test-bucket", prefix="eval-artifacts/dev", client=client)

    import_raw_dataset_from_local_dir(
        store,
        "raw_s3_output_001",
        source_dir,
        dataset_name="demo-output",
    )

    assert store.is_complete(RAW_DATASET_ARTIFACT_TYPE, "raw_s3_output_001") is True
    loaded = read_raw_dataset_artifact(store, "raw_s3_output_001")
    assert loaded.dataset_name == "demo-output"
