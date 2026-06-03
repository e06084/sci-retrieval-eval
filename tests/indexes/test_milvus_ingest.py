"""Tests for Milvus ingest from aligned chunk and embedding artifacts."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import eval_platform.indexes.milvus as milvus_module
from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    ChunkedCorpus,
    ChunkRecord,
    ProgressEvent,
    write_chunked_corpus_artifact,
)
from eval_platform.embeddings import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EmbeddedCorpus,
    EmbeddingProvenance,
    EmbeddingRecord,
    EmbeddingShard,
    write_embeddings_artifact,
)
from eval_platform.indexes import (
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    MilvusIngestConfig,
    MilvusIngestError,
    MilvusInsertFailure,
    MilvusInsertResult,
    MilvusRow,
    PymilvusMilvusClient,
    PymilvusMilvusClientConfig,
    default_milvus_schema,
    run_milvus_ingest,
)


class FakeMilvusClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        fail_insert: bool = False,
        count_override: int | None = None,
    ) -> None:
        self.exists = exists
        self.fail_insert = fail_insert
        self.count_override = count_override
        self.created_collections: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.dropped_collections: list[str] = []
        self.insert_calls: list[list[MilvusRow]] = []
        self.flushed_collections: list[str] = []
        self.rows: dict[str, dict[str, Any]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(
        self,
        collection_name: str,
        schema: dict[str, Any],
        index_params: dict[str, Any],
    ) -> None:
        self.created_collections.append((collection_name, schema, index_params))
        self.exists = True

    def drop_collection(self, collection_name: str) -> None:
        self.dropped_collections.append(collection_name)
        self.rows.clear()
        self.exists = False

    def insert_rows(
        self,
        collection_name: str,
        rows: Sequence[MilvusRow],
    ) -> MilvusInsertResult:
        self.insert_calls.append(list(rows))
        if self.fail_insert:
            return MilvusInsertResult(
                inserted_count=max(0, len(rows) - 1),
                failed_items=[
                    MilvusInsertFailure(
                        primary_key=rows[0].primary_key,
                        error="simulated failure",
                    )
                ],
            )
        for row in rows:
            self.rows[row.primary_key] = row.row
        return MilvusInsertResult(inserted_count=len(rows))

    def flush_collection(self, collection_name: str) -> None:
        self.flushed_collections.append(collection_name)

    def count_entities(self, collection_name: str) -> int:
        if self.count_override is not None:
            return self.count_override
        return len(self.rows)


def _sample_chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="chunk-1",
            doc_id="doc-1",
            title="Doc 1",
            text="first text",
            chunk_index=0,
            start_offset=0,
            end_offset=10,
            metadata={"section": "abstract"},
        ),
        ChunkRecord(
            chunk_id="chunk-2",
            doc_id="doc-1",
            title="Doc 1",
            text="second text",
            chunk_index=1,
            metadata={"section": "body"},
        ),
        ChunkRecord(
            chunk_id="chunk-3",
            doc_id="doc-2",
            text="third text",
            chunk_index=0,
        ),
    ]


def _sample_embeddings() -> list[EmbeddingRecord]:
    return [
        EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
        EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.3, 0.4]),
        EmbeddingRecord(chunk_id="chunk-3", doc_id="doc-2", vector=[0.5, 0.6]),
    ]


def _write_chunk_artifact(store: LocalArtifactStore) -> None:
    write_chunked_corpus_artifact(
        store,
        "litsearch_chunks",
        ChunkedCorpus(chunks=_sample_chunks()),
        file_record_num=1,
    )


def _write_embedding_artifact(
    store: LocalArtifactStore,
    *,
    embeddings: list[EmbeddingRecord] | None = None,
    shards: list[EmbeddingShard] | None = None,
    embedding_dim: int = 2,
) -> None:
    embedding_records = embeddings or _sample_embeddings()
    if shards is None:
        shards = [
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                first_chunk_id=embedding_records[0].chunk_id,
                last_chunk_id=embedding_records[1].chunk_id,
                embeddings=embedding_records[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                first_chunk_id=embedding_records[2].chunk_id,
                last_chunk_id=embedding_records[2].chunk_id,
                embeddings=embedding_records[2:3],
            ),
        ]

    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        EmbeddedCorpus(embeddings=embedding_records),
        provenance=EmbeddingProvenance(
            model_name="fake-embedding-model",
            embedding_dim=embedding_dim,
            normalized=True,
        ),
        source_artifact_id="litsearch_chunks",
        shards=shards,
    )


@pytest.fixture
def chunk_store(tmp_path: Path) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp_path / "chunks")
    _write_chunk_artifact(store)
    return store


@pytest.fixture
def embedding_store(tmp_path: Path) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp_path / "embeddings")
    _write_embedding_artifact(store)
    return store


@pytest.fixture
def output_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "output")


def _config(**overrides: Any) -> MilvusIngestConfig:
    payload: dict[str, Any] = {
        "chunked_corpus_artifact_id": "litsearch_chunks",
        "embeddings_artifact_id": "litsearch_embeddings",
        "output_artifact_id": "litsearch_milvus_collection",
        "collection_name": "litsearch_milvus",
        "batch_size": 2,
    }
    payload.update(overrides)
    return MilvusIngestConfig(**payload)


def _assert_no_success(output_store: LocalArtifactStore) -> None:
    assert (
        output_store.is_complete(
            MILVUS_COLLECTION_ARTIFACT_TYPE,
            "litsearch_milvus_collection",
        )
        is False
    )


def test_run_milvus_ingest_streams_aligned_shards_to_rows(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient()

    manifest = run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    assert manifest.artifact_type == MILVUS_COLLECTION_ARTIFACT_TYPE
    assert [row.primary_key for call in client.insert_calls for row in call] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    first_row = client.insert_calls[0][0].row
    assert first_row["chunk_id"] == "chunk-1"
    assert first_row["doc_id"] == "doc-1"
    assert first_row["text"] == "first text"
    assert first_row["metadata"] == {"section": "abstract"}
    assert first_row["source_chunked_corpus_artifact_id"] == "litsearch_chunks"
    assert first_row["source_embeddings_artifact_id"] == "litsearch_embeddings"
    assert first_row["source_chunk_file"] == "chunks/part-00000.jsonl"
    assert first_row["source_embedding_file"] == "embeddings/part-00000.jsonl"
    assert first_row["shard_id"] == "part-00000"
    assert first_row["vector"] == pytest.approx([0.1, 0.2])


def test_run_milvus_ingest_uses_chunk_id_as_primary_key(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient()

    run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    assert set(client.rows) == {"chunk-1", "chunk-2", "chunk-3"}


def test_run_milvus_ingest_batch_size_preserves_source_order(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient()

    run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(batch_size=1),
        client,
    )

    assert [[row.primary_key for row in call] for call in client.insert_calls] == [
        ["chunk-1"],
        ["chunk-2"],
        ["chunk-3"],
    ]


def test_run_milvus_ingest_creates_collection_with_default_schema(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient()

    run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    assert client.created_collections == [
        (
            "litsearch_milvus",
            default_milvus_schema(vector_dim=2),
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
    ]


def test_default_milvus_schema_uses_sciverse_v1_field_defaults() -> None:
    schema = default_milvus_schema(vector_dim=7)

    fields = {field["name"]: field for field in schema["fields"]}
    assert fields["chunk_id"]["is_primary"] is True
    assert fields["vector"]["dtype"] == "FLOAT_VECTOR"
    assert fields["vector"]["dim"] == 7
    assert fields["title"]["max_length"] == 65535
    assert fields["text"]["max_length"] == 65535


def test_run_milvus_ingest_allows_explicit_index_params_override(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient()

    run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(index_params={"index_type": "AUTOINDEX", "metric_type": "IP"}),
        client,
    )

    assert client.created_collections[0][2] == {
        "index_type": "AUTOINDEX",
        "metric_type": "IP",
        "params": {},
    }


def test_run_milvus_ingest_rejects_existing_collection_without_overwrite(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient(exists=True)

    with pytest.raises(MilvusIngestError, match="already exists"):
        run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    _assert_no_success(output_store)


def test_run_milvus_ingest_overwrite_drops_then_creates(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient(exists=True)

    run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(overwrite_existing=True),
        client,
    )

    assert client.dropped_collections == ["litsearch_milvus"]
    assert client.created_collections[0][0] == "litsearch_milvus"


def test_run_milvus_ingest_insert_failure_does_not_write_success(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient(fail_insert=True)

    with pytest.raises(MilvusIngestError, match="insert failed"):
        run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    _assert_no_success(output_store)


def test_run_milvus_ingest_count_mismatch_does_not_write_success(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeMilvusClient(count_override=2)

    with pytest.raises(MilvusIngestError, match="count verification failed"):
        run_milvus_ingest(chunk_store, embedding_store, output_store, _config(), client)

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_shard_count_mismatch(
    chunk_store: LocalArtifactStore,
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    bad_embedding_store = LocalArtifactStore(tmp_path / "bad_embeddings")
    embeddings = [
        *_sample_embeddings(),
        EmbeddingRecord(chunk_id="chunk-4", doc_id="doc-4", vector=[0.7, 0.8]),
    ]
    _write_embedding_artifact(
        bad_embedding_store,
        embeddings=embeddings,
        shards=[
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                first_chunk_id="chunk-1",
                last_chunk_id="chunk-2",
                embeddings=embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                first_chunk_id="chunk-3",
                last_chunk_id="chunk-3",
                embeddings=embeddings[2:3],
            ),
            EmbeddingShard(
                shard_id="part-00002",
                source_chunk_file="chunks/part-00002.jsonl",
                embedding_file="embeddings/part-00002.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                first_chunk_id="chunk-4",
                last_chunk_id="chunk-4",
                embeddings=embeddings[3:],
            ),
        ],
    )

    with pytest.raises(MilvusIngestError, match="shard counts do not match"):
        run_milvus_ingest(
            chunk_store,
            bad_embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_shard_id_mismatch(
    chunk_store: LocalArtifactStore,
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    bad_embedding_store = LocalArtifactStore(tmp_path / "bad_embeddings")
    embeddings = _sample_embeddings()
    _write_embedding_artifact(
        bad_embedding_store,
        embeddings=embeddings,
        shards=[
            EmbeddingShard(
                shard_id="part-99999",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                embeddings=embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                embeddings=embeddings[2:3],
            ),
        ],
    )

    with pytest.raises(MilvusIngestError, match="shard_id mismatch"):
        run_milvus_ingest(
            chunk_store,
            bad_embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_source_chunk_file_mismatch(
    chunk_store: LocalArtifactStore,
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    bad_embedding_store = LocalArtifactStore(tmp_path / "bad_embeddings")
    embeddings = _sample_embeddings()
    _write_embedding_artifact(
        bad_embedding_store,
        embeddings=embeddings,
        shards=[
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/wrong.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                embeddings=embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                embeddings=embeddings[2:3],
            ),
        ],
    )

    with pytest.raises(MilvusIngestError, match="source_chunk_file"):
        run_milvus_ingest(
            chunk_store,
            bad_embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_chunk_id_mismatch(
    chunk_store: LocalArtifactStore,
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    bad_embedding_store = LocalArtifactStore(tmp_path / "bad_embeddings")
    embeddings = _sample_embeddings()
    embeddings[1] = EmbeddingRecord(chunk_id="wrong", doc_id="doc-1", vector=[0.3, 0.4])
    _write_embedding_artifact(bad_embedding_store, embeddings=embeddings)

    with pytest.raises(MilvusIngestError, match="chunk_id mismatch"):
        run_milvus_ingest(
            chunk_store,
            bad_embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_doc_id_mismatch(
    chunk_store: LocalArtifactStore,
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    bad_embedding_store = LocalArtifactStore(tmp_path / "bad_embeddings")
    embeddings = _sample_embeddings()
    embeddings[1] = EmbeddingRecord(chunk_id="chunk-2", doc_id="wrong", vector=[0.3, 0.4])
    _write_embedding_artifact(bad_embedding_store, embeddings=embeddings)

    with pytest.raises(MilvusIngestError, match="doc_id mismatch"):
        run_milvus_ingest(
            chunk_store,
            bad_embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_rejects_vector_dim_mismatch(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = embedding_store.read_manifest(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings")
    manifest.metadata["embedding_dim"] = 3
    embedding_store.write_manifest(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings", manifest)

    with pytest.raises(MilvusIngestError, match="dimension mismatch"):
        run_milvus_ingest(
            chunk_store,
            embedding_store,
            output_store,
            _config(vector_dim=3),
            FakeMilvusClient(),
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_writes_manifest_dependencies_and_metadata(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config(
        metadata={
            "inserted_count": 999,
            "schema_sha256": "wrong",
            "password": "should-not-be-system",
            "api_token": "also-secret",
            "stage": "milvus",
        }
    )

    manifest = run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        config,
        FakeMilvusClient(),
    )

    assert output_store.is_complete(
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        "litsearch_milvus_collection",
    )
    assert [dependency.artifact_type for dependency in manifest.dependencies] == [
        "chunked_corpus",
        "embeddings",
    ]
    assert [dependency.artifact_id for dependency in manifest.dependencies] == [
        "litsearch_chunks",
        "litsearch_embeddings",
    ]
    assert manifest.files == []
    assert manifest.metadata["source_chunked_corpus_artifact_id"] == "litsearch_chunks"
    assert manifest.metadata["source_embeddings_artifact_id"] == "litsearch_embeddings"
    assert manifest.metadata["collection_name"] == "litsearch_milvus"
    assert manifest.metadata["primary_key_field"] == "chunk_id"
    assert manifest.metadata["vector_field"] == "vector"
    assert manifest.metadata["vector_dim"] == 2
    assert manifest.metadata["metric_type"] == "COSINE"
    assert manifest.metadata["index_params"] == {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    assert manifest.metadata["inserted_count"] == 3
    assert manifest.metadata["failed_count"] == 0
    assert manifest.metadata["verified_entity_count"] == 3
    assert manifest.metadata["schema_sha256"] != "wrong"
    assert manifest.metadata["alignment_key"] == "chunk_id"
    assert manifest.metadata["alignment_order"] == "source_chunk_order"
    assert len(manifest.metadata["shards"]) == 2
    assert manifest.metadata["shards"][0]["source_chunk_file"] == "chunks/part-00000.jsonl"
    assert manifest.metadata["shards"][0]["source_embedding_file"] == (
        "embeddings/part-00000.jsonl"
    )
    assert manifest.metadata["shards"][0]["inserted_count"] == 2
    assert manifest.metadata["shards"][1]["inserted_count"] == 1
    assert manifest.metadata["stage"] == "milvus"
    assert "password" not in manifest.metadata
    assert "api_token" not in manifest.metadata


def test_milvus_fingerprint_components_record_effective_index_params() -> None:
    components = milvus_module._milvus_asset_fingerprint_components(
        config=_config(),
        chunked_corpus_fingerprint="chunk-fp",
        embeddings_fingerprint="embedding-fp",
        schema=default_milvus_schema(vector_dim=2),
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )

    assert components is not None
    assert components["index_type"] == "HNSW"
    assert components["metric_type"] == "COSINE"
    assert components["index_params"] == {
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }


def test_run_milvus_ingest_manifest_does_not_record_connection_secrets(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(),
        FakeMilvusClient(),
    )
    manifest_text = str(manifest.model_dump(mode="json")).lower()

    assert "password" not in manifest_text
    assert "token" not in manifest_text
    assert "access_key" not in manifest_text


def test_run_milvus_ingest_reports_progress(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    events: list[ProgressEvent] = []

    run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(),
        FakeMilvusClient(),
        progress_reporter=events.append,
    )

    assert "collection" in [event.metadata.get("kind") for event in events]
    assert "batch" in [event.metadata.get("kind") for event in events]
    assert "shard" in [event.metadata.get("kind") for event in events]
    assert "flush" in [event.metadata.get("kind") for event in events]
    assert "verify" in [event.metadata.get("kind") for event in events]


def test_run_milvus_ingest_progress_failure_does_not_write_success(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    def fail_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        run_milvus_ingest(
            chunk_store,
            embedding_store,
            output_store,
            _config(),
            FakeMilvusClient(),
            progress_reporter=fail_reporter,
        )

    _assert_no_success(output_store)


def test_run_milvus_ingest_does_not_use_full_artifact_readers(
    chunk_store: LocalArtifactStore,
    embedding_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("full artifact reader must not be used")

    monkeypatch.setattr(milvus_module, "read_chunked_corpus_artifact", fail_read, raising=False)
    monkeypatch.setattr(milvus_module, "read_embeddings_artifact", fail_read, raising=False)

    run_milvus_ingest(
        chunk_store,
        embedding_store,
        output_store,
        _config(),
        FakeMilvusClient(),
    )

    assert output_store.is_complete(
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        "litsearch_milvus_collection",
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunked_corpus_artifact_id", ""),
        ("chunked_corpus_artifact_id", " "),
        ("embeddings_artifact_id", ""),
        ("embeddings_artifact_id", " "),
        ("output_artifact_id", ""),
        ("output_artifact_id", " "),
        ("collection_name", ""),
        ("collection_name", " "),
        ("primary_key_field", ""),
        ("primary_key_field", " "),
        ("vector_field", ""),
        ("vector_field", " "),
        ("metric_type", ""),
        ("metric_type", " "),
    ],
)
def test_milvus_ingest_config_rejects_blank_strings(field_name: str, value: str) -> None:
    payload = {
        "chunked_corpus_artifact_id": "chunks",
        "embeddings_artifact_id": "embeddings",
        "output_artifact_id": "milvus",
        "collection_name": "collection",
        field_name: value,
    }

    with pytest.raises(ValidationError):
        MilvusIngestConfig.model_validate(payload)


def test_milvus_ingest_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValidationError):
        MilvusIngestConfig(
            chunked_corpus_artifact_id="chunks",
            embeddings_artifact_id="embeddings",
            output_artifact_id="milvus",
            collection_name="collection",
            batch_size=0,
        )


def test_milvus_ingest_config_rejects_non_positive_vector_dim() -> None:
    with pytest.raises(ValidationError):
        MilvusIngestConfig(
            chunked_corpus_artifact_id="chunks",
            embeddings_artifact_id="embeddings",
            output_artifact_id="milvus",
            collection_name="collection",
            vector_dim=0,
        )


def test_pymilvus_client_converts_dict_schema_and_index_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataType:
        VARCHAR = "VARCHAR"
        INT64 = "INT64"
        JSON = "JSON"
        FLOAT_VECTOR = "FLOAT_VECTOR"

    class FakeSchema:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.fields: list[tuple[str, str, dict[str, Any]]] = []

        def add_field(self, field_name: str, datatype: str, **kwargs: Any) -> None:
            self.fields.append((field_name, datatype, kwargs))

    class FakeIndexParams:
        def __init__(self) -> None:
            self.indexes: list[dict[str, Any]] = []

        def add_index(self, **kwargs: Any) -> None:
            self.indexes.append(kwargs)

    class FakeMilvusClientFactory:
        @staticmethod
        def create_schema(**kwargs: Any) -> FakeSchema:
            return FakeSchema(**kwargs)

        @staticmethod
        def prepare_index_params() -> FakeIndexParams:
            return FakeIndexParams()

    class RecordingClient:
        def __init__(self) -> None:
            self.created: tuple[str, FakeSchema, FakeIndexParams] | None = None

        def create_collection(
            self,
            *,
            collection_name: str,
            schema: FakeSchema,
            index_params: FakeIndexParams,
        ) -> None:
            self.created = (collection_name, schema, index_params)

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(DataType=FakeDataType, MilvusClient=FakeMilvusClientFactory),
    )
    recording_client = RecordingClient()
    client = PymilvusMilvusClient(
        PymilvusMilvusClientConfig(uri="http://milvus.example:19530"),
        client=recording_client,
    )

    client.create_collection(
        "collection",
        default_milvus_schema(vector_dim=2),
        {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8}},
    )

    assert recording_client.created is not None
    collection_name, schema, index_params = recording_client.created
    assert collection_name == "collection"
    assert ("vector", "FLOAT_VECTOR", {"dim": 2}) in schema.fields
    assert index_params.indexes == [
        {
            "field_name": "vector",
            "index_type": "HNSW",
            "index_name": "",
            "metric_type": "COSINE",
            "params": {"M": 8},
        }
    ]
