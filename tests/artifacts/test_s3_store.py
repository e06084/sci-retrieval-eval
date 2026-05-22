"""Tests for S3ArtifactStore."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from eval_platform.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactNotFoundError,
    InvalidArtifactPathError,
    ManifestNotFoundError,
    S3ArtifactStore,
)


class FakeObjectNotFoundError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeNoSuchBucketError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchBucket"}}


class FakeS3ClientMissingBucket:
    def head_object(self, *, Bucket: str, Key: str) -> None:
        raise FakeNoSuchBucketError


class FakeS3Client:
    def __init__(self, page_size: int | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size

    def _full_key(self, bucket: str, key: str) -> str:
        return f"{bucket}/{key}"

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | io.BytesIO) -> None:
        data = Body.read() if hasattr(Body, "read") else Body
        self.objects[self._full_key(Bucket, Key)] = data

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        full_key = self._full_key(Bucket, Key)
        if full_key not in self.objects:
            raise FakeObjectNotFoundError
        return {"Body": io.BytesIO(self.objects[full_key])}

    def head_object(self, *, Bucket: str, Key: str) -> None:
        full_key = self._full_key(Bucket, Key)
        if full_key not in self.objects:
            raise FakeObjectNotFoundError

    def _matching_keys(self, bucket: str, prefix: str) -> list[str]:
        bucket_prefix = f"{bucket}/"
        keys: list[str] = []
        for full_key in self.objects:
            if not full_key.startswith(bucket_prefix):
                continue
            key = full_key[len(bucket_prefix) :]
            if prefix and not key.startswith(prefix):
                continue
            keys.append(key)
        return sorted(keys)

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        ContinuationToken: str | None = None,
        MaxKeys: int | None = None,
    ) -> dict[str, object]:
        matching_keys = self._matching_keys(Bucket, Prefix)
        page_size = MaxKeys or self.page_size or len(matching_keys)
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
def bucket() -> str:
    return "test-bucket"


@pytest.fixture
def client() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def store(bucket: str, client: FakeS3Client) -> S3ArtifactStore:
    return S3ArtifactStore(bucket=bucket, prefix="eval-artifacts/dev", client=client)


def _sample_manifest(artifact_id: str = "sample_001") -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type="normalized_dataset",
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
        created_by="test",
        files=[ArtifactFile(path="corpus.jsonl", size_bytes=128, sha256="abc123")],
    )


def test_put_get_and_exists(store: S3ArtifactStore) -> None:
    store.put_file("normalized_dataset", "sample_001", "corpus.jsonl", b'{"id": 1}\n')

    assert store.exists("normalized_dataset", "sample_001", "corpus.jsonl") is True
    assert store.get_file("normalized_dataset", "sample_001", "corpus.jsonl") == b'{"id": 1}\n'


def test_write_and_read_manifest(store: S3ArtifactStore) -> None:
    manifest = _sample_manifest()

    store.write_manifest("normalized_dataset", "sample_001", manifest)
    loaded = store.read_manifest("normalized_dataset", "sample_001")

    assert loaded == manifest


def test_is_complete_false_with_manifest_only(store: S3ArtifactStore) -> None:
    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest())

    assert store.is_complete("normalized_dataset", "sample_001") is False


def test_is_complete_false_with_success_only(store: S3ArtifactStore) -> None:
    store.mark_success("normalized_dataset", "sample_001")

    assert store.is_complete("normalized_dataset", "sample_001") is False


def test_is_complete_true_with_manifest_and_success(store: S3ArtifactStore) -> None:
    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest())
    store.mark_success("normalized_dataset", "sample_001")

    assert store.is_complete("normalized_dataset", "sample_001") is True


def test_read_manifest_raises_when_missing(store: S3ArtifactStore) -> None:
    with pytest.raises(ManifestNotFoundError):
        store.read_manifest("normalized_dataset", "missing")


def test_get_file_raises_when_missing(store: S3ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        store.get_file("normalized_dataset", "sample_001", "missing.jsonl")


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.txt", "/tmp/evil.txt", "nested/../../outside.txt"],
)
def test_rejects_unsafe_relative_path(store: S3ArtifactStore, relative_path: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file("normalized_dataset", "sample_001", relative_path, b"data")


@pytest.mark.parametrize(
    "artifact_type",
    ["../escape", "../../x", "bad/type"],
)
def test_rejects_unsafe_artifact_type(store: S3ArtifactStore, artifact_type: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file(artifact_type, "sample_001", "corpus.jsonl", b"data")


@pytest.mark.parametrize(
    "artifact_id",
    ["../escape", "../../y", "bad/id"],
)
def test_rejects_unsafe_artifact_id(store: S3ArtifactStore, artifact_id: str) -> None:
    with pytest.raises(InvalidArtifactPathError):
        store.put_file("normalized_dataset", artifact_id, "corpus.jsonl", b"data")


def test_artifact_uri_with_prefix(bucket: str, client: FakeS3Client) -> None:
    store = S3ArtifactStore(bucket=bucket, prefix="eval-artifacts/dev", client=client)

    assert (
        store.artifact_uri("normalized_dataset", "litsearch_001")
        == f"s3://{bucket}/eval-artifacts/dev/normalized_dataset/litsearch_001/"
    )


def test_artifact_uri_without_prefix(bucket: str, client: FakeS3Client) -> None:
    store = S3ArtifactStore(bucket=bucket, prefix="", client=client)

    assert (
        store.artifact_uri("normalized_dataset", "litsearch_001")
        == f"s3://{bucket}/normalized_dataset/litsearch_001/"
    )


def test_list_artifacts(store: S3ArtifactStore) -> None:
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


def test_list_artifacts_handles_pagination(bucket: str) -> None:
    client = FakeS3Client(page_size=1)
    store = S3ArtifactStore(bucket=bucket, prefix="eval-artifacts/dev", client=client)

    store.write_manifest("normalized_dataset", "sample_001", _sample_manifest("sample_001"))
    store.put_file("normalized_dataset", "sample_001", "extra.jsonl", b"{}\n")
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


def test_no_such_bucket_is_not_silently_ignored(bucket: str) -> None:
    client = FakeS3ClientMissingBucket()
    store = S3ArtifactStore(bucket=bucket, prefix="", client=client)

    with pytest.raises(FakeNoSuchBucketError):
        store.exists("normalized_dataset", "sample_001", "corpus.jsonl")

    with pytest.raises(FakeNoSuchBucketError):
        store.is_complete("normalized_dataset", "sample_001")


def test_default_client_requires_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="boto3 is required"):
        S3ArtifactStore(bucket="test-bucket")
