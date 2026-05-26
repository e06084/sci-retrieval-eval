"""Tests for Elasticsearch ingest from chunked corpus artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import eval_platform.indexes.elasticsearch as es_module
from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    ChunkedCorpus,
    ChunkRecord,
    ProgressEvent,
    write_chunked_corpus_artifact,
)
from eval_platform.indexes import (
    DEFAULT_ELASTICSEARCH_MAPPING,
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    ElasticsearchBulkAction,
    ElasticsearchBulkFailure,
    ElasticsearchBulkResult,
    ElasticsearchIngestConfig,
    ElasticsearchIngestError,
    run_elasticsearch_ingest,
)


class FakeElasticsearchClient:
    def __init__(
        self,
        *,
        exists: bool = False,
        fail_bulk: bool = False,
        count_override: int | None = None,
    ) -> None:
        self.exists = exists
        self.fail_bulk = fail_bulk
        self.count_override = count_override
        self.created_indices: list[tuple[str, dict[str, Any]]] = []
        self.deleted_indices: list[str] = []
        self.bulk_calls: list[list[ElasticsearchBulkAction]] = []
        self.refreshed_indices: list[str] = []
        self.documents: dict[str, dict[str, Any]] = {}

    def index_exists(self, index_name: str) -> bool:
        return self.exists

    def create_index(self, index_name: str, body: dict[str, Any]) -> None:
        self.created_indices.append((index_name, body))
        self.exists = True

    def delete_index(self, index_name: str) -> None:
        self.deleted_indices.append(index_name)
        self.documents.clear()
        self.exists = False

    def bulk_index(
        self,
        index_name: str,
        actions: Sequence[ElasticsearchBulkAction],
    ) -> ElasticsearchBulkResult:
        self.bulk_calls.append(list(actions))
        if self.fail_bulk:
            return ElasticsearchBulkResult(
                indexed_count=max(0, len(actions) - 1),
                failed_items=[
                    ElasticsearchBulkFailure(
                        document_id=actions[0].document_id,
                        status=500,
                        error="simulated failure",
                    )
                ],
            )
        for action in actions:
            self.documents[action.document_id] = action.document
        return ElasticsearchBulkResult(indexed_count=len(actions))

    def refresh_index(self, index_name: str) -> None:
        self.refreshed_indices.append(index_name)

    def count_documents(self, index_name: str) -> int:
        if self.count_override is not None:
            return self.count_override
        return len(self.documents)


@pytest.fixture
def source_store(tmp_path: Path) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp_path / "source")
    write_chunked_corpus_artifact(
        store,
        "litsearch_chunks",
        ChunkedCorpus(
            chunks=[
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
        ),
        file_record_num=1,
    )
    return store


@pytest.fixture
def output_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "output")


def _config(**overrides: Any) -> ElasticsearchIngestConfig:
    payload: dict[str, Any] = {
        "source_artifact_id": "litsearch_chunks",
        "output_artifact_id": "litsearch_es_index",
        "index_name": "litsearch-test",
        "bulk_size": 2,
    }
    payload.update(overrides)
    return ElasticsearchIngestConfig(**payload)


def test_run_elasticsearch_ingest_streams_sharded_chunks_to_bulk_actions(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient()

    manifest = run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert manifest.artifact_type == ELASTICSEARCH_INDEX_ARTIFACT_TYPE
    assert [action.document_id for call in client.bulk_calls for action in call] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert client.bulk_calls[0][0].document["source_chunk_file"] == "chunks/part-00000.jsonl"
    assert client.bulk_calls[0][0].document["shard_id"] == "part-00000"
    assert client.bulk_calls[0][0].document["source_chunked_corpus_artifact_id"] == (
        "litsearch_chunks"
    )
    assert client.bulk_calls[0][0].document["metadata"] == {"section": "abstract"}


def test_run_elasticsearch_ingest_uses_chunk_id_as_document_id(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient()

    run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert set(client.documents) == {"chunk-1", "chunk-2", "chunk-3"}


def test_run_elasticsearch_ingest_bulk_size_preserves_source_order(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient()

    run_elasticsearch_ingest(source_store, output_store, _config(bulk_size=1), client)

    assert [[action.document_id for action in call] for call in client.bulk_calls] == [
        ["chunk-1"],
        ["chunk-2"],
        ["chunk-3"],
    ]


def test_run_elasticsearch_ingest_creates_index_with_default_mapping(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient()

    run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert client.created_indices == [("litsearch-test", DEFAULT_ELASTICSEARCH_MAPPING)]


def test_run_elasticsearch_ingest_rejects_existing_index_without_overwrite(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient(exists=True)

    with pytest.raises(ElasticsearchIngestError, match="already exists"):
        run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert (
        output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")
        is False
    )


def test_run_elasticsearch_ingest_overwrite_deletes_then_creates(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient(exists=True)

    run_elasticsearch_ingest(
        source_store,
        output_store,
        _config(overwrite_existing=True),
        client,
    )

    assert client.deleted_indices == ["litsearch-test"]
    assert client.created_indices[0][0] == "litsearch-test"


def test_run_elasticsearch_ingest_bulk_failure_does_not_write_success(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient(fail_bulk=True)

    with pytest.raises(ElasticsearchIngestError, match="bulk index failed"):
        run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert (
        output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")
        is False
    )


def test_run_elasticsearch_ingest_count_mismatch_does_not_write_success(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient(count_override=2)

    with pytest.raises(ElasticsearchIngestError, match="count verification failed"):
        run_elasticsearch_ingest(source_store, output_store, _config(), client)

    assert (
        output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")
        is False
    )


def test_run_elasticsearch_ingest_writes_manifest_dependency_and_metadata(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    client = FakeElasticsearchClient()
    config = _config(
        metadata={
            "indexed_count": 999,
            "mapping_sha256": "wrong",
            "password": "should-not-be-system",
            "api_token": "also-secret",
            "stage": "es",
        }
    )

    manifest = run_elasticsearch_ingest(source_store, output_store, config, client)

    assert output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")
    assert manifest.dependencies[0].artifact_type == "chunked_corpus"
    assert manifest.dependencies[0].artifact_id == "litsearch_chunks"
    assert manifest.files == []
    assert manifest.metadata["source_chunked_corpus_artifact_id"] == "litsearch_chunks"
    assert manifest.metadata["index_name"] == "litsearch-test"
    assert manifest.metadata["document_id_field"] == "chunk_id"
    assert manifest.metadata["indexed_count"] == 3
    assert manifest.metadata["failed_count"] == 0
    assert manifest.metadata["verified_document_count"] == 3
    assert manifest.metadata["mapping_sha256"] != "wrong"
    assert "password" not in manifest.metadata
    assert "api_token" not in manifest.metadata
    assert len(manifest.metadata["shards"]) == 2
    assert manifest.metadata["shards"][0]["source_chunk_file"] == "chunks/part-00000.jsonl"
    assert manifest.metadata["shards"][0]["indexed_count"] == 2
    assert manifest.metadata["shards"][1]["indexed_count"] == 1
    assert manifest.metadata["stage"] == "es"


def test_run_elasticsearch_ingest_manifest_does_not_record_connection_secrets(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_elasticsearch_ingest(
        source_store,
        output_store,
        _config(),
        FakeElasticsearchClient(),
    )
    manifest_text = str(manifest.model_dump(mode="json")).lower()

    assert "password" not in manifest_text
    assert "token" not in manifest_text
    assert "access_key" not in manifest_text


def test_run_elasticsearch_ingest_reports_progress(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    events: list[ProgressEvent] = []

    run_elasticsearch_ingest(
        source_store,
        output_store,
        _config(),
        FakeElasticsearchClient(),
        progress_reporter=events.append,
    )

    assert "index" in [event.metadata.get("kind") for event in events]
    assert "bulk" in [event.metadata.get("kind") for event in events]
    assert "shard" in [event.metadata.get("kind") for event in events]
    assert "verify" in [event.metadata.get("kind") for event in events]


def test_run_elasticsearch_ingest_progress_failure_does_not_write_success(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    def fail_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        run_elasticsearch_ingest(
            source_store,
            output_store,
            _config(),
            FakeElasticsearchClient(),
            progress_reporter=fail_reporter,
        )

    assert (
        output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")
        is False
    )


def test_run_elasticsearch_ingest_does_not_use_full_chunk_reader(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("full chunk reader must not be used")

    monkeypatch.setattr(es_module, "read_chunked_corpus_artifact", fail_read, raising=False)

    run_elasticsearch_ingest(source_store, output_store, _config(), FakeElasticsearchClient())

    assert output_store.is_complete(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, "litsearch_es_index")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_artifact_id", ""),
        ("source_artifact_id", " "),
        ("output_artifact_id", ""),
        ("output_artifact_id", " "),
        ("index_name", ""),
        ("index_name", " "),
    ],
)
def test_elasticsearch_ingest_config_rejects_blank_strings(
    field_name: str,
    value: str,
) -> None:
    payload = {
        "source_artifact_id": "chunks",
        "output_artifact_id": "index",
        "index_name": "index-name",
        field_name: value,
    }

    with pytest.raises(ValidationError):
        ElasticsearchIngestConfig.model_validate(payload)


def test_elasticsearch_ingest_config_rejects_non_positive_bulk_size() -> None:
    with pytest.raises(ValidationError):
        ElasticsearchIngestConfig(
            source_artifact_id="chunks",
            output_artifact_id="index",
            index_name="index-name",
            bulk_size=0,
        )
