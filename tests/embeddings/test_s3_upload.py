"""Tests for embedding runner S3 upload behavior."""

import io
from pathlib import Path

from eval_platform.artifacts import LocalArtifactStore, S3ArtifactStore
from eval_platform.chunking import ChunkRecord, write_chunked_corpus_artifact
from eval_platform.chunking.schema import ChunkedCorpus
from eval_platform.embeddings import (
    EmbeddingRunConfig,
    FakeEmbeddingClient,
    read_embeddings_artifact,
    run_embedding,
)


class FakeObjectNotFoundError(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

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

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        ContinuationToken: str | None = None,
    ) -> dict[str, object]:
        bucket_prefix = f"{Bucket}/"
        keys = sorted(
            key[len(bucket_prefix) :]
            for key in self.objects
            if key.startswith(bucket_prefix) and key[len(bucket_prefix) :].startswith(Prefix)
        )
        return {
            "Contents": [{"Key": key} for key in keys],
            "IsTruncated": False,
            "NextContinuationToken": None,
        }


def test_run_embedding_local_source_to_s3_output(tmp_path: Path) -> None:
    local_store = LocalArtifactStore(tmp_path / "local")
    write_chunked_corpus_artifact(
        local_store,
        "litsearch_chunks",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(chunk_id="chunk-1", doc_id="doc-1", text="first", chunk_index=0),
                ChunkRecord(chunk_id="chunk-2", doc_id="doc-2", text="second", chunk_index=0),
            ]
        ),
    )

    s3_store = S3ArtifactStore(
        bucket="test-bucket",
        prefix="eval-artifacts/dev",
        client=FakeS3Client(),
    )
    config = EmbeddingRunConfig(
        source_artifact_id="litsearch_chunks",
        output_artifact_id="litsearch_embeddings",
        model_name="fake-embedding-model",
        embedding_dim=3,
        provider="fake",
        normalized=True,
    )

    manifest = run_embedding(local_store, s3_store, config, FakeEmbeddingClient(3))

    assert s3_store.is_complete("embeddings", "litsearch_embeddings") is True
    assert s3_store.exists("embeddings", "litsearch_embeddings", "embeddings.jsonl") is True
    assert s3_store.exists("embeddings", "litsearch_embeddings", "_MANIFEST.json") is True
    assert s3_store.exists("embeddings", "litsearch_embeddings", "_SUCCESS") is True
    loaded = read_embeddings_artifact(s3_store, "litsearch_embeddings")
    assert len(loaded.embeddings) == 2
    assert manifest.dependencies[0].artifact_id == "litsearch_chunks"
    assert manifest.dependencies[0].artifact_type == "chunked_corpus"
    assert manifest.metadata["provenance"]["model_name"] == "fake-embedding-model"
    assert manifest.metadata["embedding_dim"] == 3
