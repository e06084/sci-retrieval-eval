"""Tests for embedding runner orchestration."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

import eval_platform.embeddings.runner as runner_module
from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    ChunkRecord,
    ProgressEvent,
    iter_chunk_shards,
    write_chunked_corpus_artifact,
)
from eval_platform.chunking.schema import ChunkedCorpus
from eval_platform.embeddings import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EMBEDDINGS_FILENAME,
    EmbeddedCorpus,
    EmbeddingConsistencyCheckResult,
    EmbeddingRecord,
    EmbeddingRunConfig,
    EmbeddingRunError,
    FakeEmbeddingClient,
    dump_embeddings_jsonl,
    iter_embedding_shards,
    read_embeddings_artifact,
    run_embedding,
)


class WrongCountEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts[:-1]]


class WrongDimEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class RecordingEmbeddingClient:
    def __init__(self, embedding_dim: int = 3) -> None:
        self._embedding_dim = embedding_dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index + 1) for index in range(self._embedding_dim)] for _ in texts]


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
                    text="first chunk text",
                    title="First",
                    chunk_index=0,
                    metadata={"section": "abstract"},
                ),
                ChunkRecord(
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    text="second chunk text",
                    chunk_index=1,
                    start_offset=10,
                    end_offset=20,
                ),
            ],
            metadata={"source": "chunking"},
        ),
    )
    return store


@pytest.fixture
def output_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "output")


def _config() -> EmbeddingRunConfig:
    return EmbeddingRunConfig(
        source_artifact_id="litsearch_chunks",
        output_artifact_id="litsearch_embeddings",
        model_name="fake-embedding-model",
        embedding_dim=3,
        provider="fake-provider",
        api_version="v1",
        normalized=True,
    )


def _write_resume_source(source_store: LocalArtifactStore) -> None:
    write_chunked_corpus_artifact(
        source_store,
        "resume_chunks",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="first chunk text",
                    chunk_index=0,
                ),
                ChunkRecord(
                    chunk_id="chunk-2",
                    doc_id="doc-1",
                    text="second chunk text",
                    chunk_index=1,
                ),
                ChunkRecord(
                    chunk_id="chunk-3",
                    doc_id="doc-2",
                    text="third chunk text",
                    chunk_index=0,
                ),
            ],
        ),
        file_record_num=1,
    )


def _resume_config(*, resume_existing_shards: bool = True) -> EmbeddingRunConfig:
    return EmbeddingRunConfig(
        source_artifact_id="resume_chunks",
        output_artifact_id="resume_embeddings",
        model_name="fake-embedding-model",
        embedding_dim=3,
        resume_existing_shards=resume_existing_shards,
    )


def _put_existing_embedding_shard(
    output_store: LocalArtifactStore,
    shard_id: str,
    records: list[EmbeddingRecord],
) -> None:
    output_store.put_file(
        EMBEDDINGS_ARTIFACT_TYPE,
        "resume_embeddings",
        f"embeddings/{shard_id}.jsonl",
        dump_embeddings_jsonl(records).encode("utf-8"),
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_artifact_id", ""),
        ("source_artifact_id", " "),
        ("output_artifact_id", ""),
        ("output_artifact_id", " "),
        ("model_name", ""),
        ("model_name", " "),
    ],
)
def test_embedding_run_config_rejects_blank_values(field_name: str, value: str) -> None:
    payload = {
        "source_artifact_id": "litsearch_chunks",
        "output_artifact_id": "litsearch_embeddings",
        "model_name": "fake-model",
        "embedding_dim": 3,
        field_name: value,
    }
    with pytest.raises(ValidationError):
        EmbeddingRunConfig.model_validate(payload)


def test_embedding_run_config_default_metadata_not_shared() -> None:
    first = EmbeddingRunConfig(
        source_artifact_id="a",
        output_artifact_id="b",
        model_name="model-a",
        embedding_dim=3,
    )
    second = EmbeddingRunConfig(
        source_artifact_id="c",
        output_artifact_id="d",
        model_name="model-b",
        embedding_dim=3,
    )
    first.metadata["run"] = 1
    second.metadata["run"] = 2
    assert first.metadata == {"run": 1}
    assert second.metadata == {"run": 2}


def test_fake_embedding_client_is_deterministic() -> None:
    client = FakeEmbeddingClient(embedding_dim=3)
    texts = ["a", "b"]
    assert client.embed_texts(texts) == client.embed_texts(texts)


def test_run_embedding_local_to_local_round_trip(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_embedding(source_store, output_store, _config(), FakeEmbeddingClient(3))
    loaded = read_embeddings_artifact(output_store, "litsearch_embeddings")

    assert manifest.artifact_id == "litsearch_embeddings"
    assert output_store.is_complete("embeddings", "litsearch_embeddings") is True
    assert isinstance(loaded, EmbeddedCorpus)
    assert len(loaded.embeddings) == 2
    assert loaded.embeddings[0].chunk_id == "chunk-1"
    assert loaded.embeddings[0].doc_id == "doc-1"
    assert loaded.embeddings[0].metadata["section"] == "abstract"
    assert loaded.embeddings[0].metadata["title"] == "First"
    assert loaded.embeddings[1].metadata["start_offset"] == 10
    assert loaded.embeddings[1].metadata["end_offset"] == 20


def test_run_embedding_records_source_dependency(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_embedding(source_store, output_store, _config(), FakeEmbeddingClient(3))
    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].artifact_id == "litsearch_chunks"
    assert manifest.dependencies[0].artifact_type == "chunked_corpus"


def test_run_embedding_records_provenance_metadata(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_id = "endpoint-a"
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]
    config.batch_size = 8
    config.timeout_seconds = 60.0
    config.consistency_check = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        passed=True,
        max_abs_diff=0.0,
    )
    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    provenance = manifest.metadata["provenance"]
    assert provenance["model_name"] == "fake-embedding-model"
    assert provenance["provider"] == "fake-provider"
    assert provenance["api_version"] == "v1"
    assert provenance["embedding_dim"] == 3
    assert provenance["normalized"] is True
    assert provenance["endpoint_id"] == "endpoint-a"
    assert provenance["endpoint_ids"] == ["endpoint-a", "endpoint-b"]
    assert provenance["consistency_check"]["passed"] is True
    assert provenance["runtime_parameters"]["batch_size"] == 8
    assert provenance["runtime_parameters"]["timeout_seconds"] == 60.0
    assert manifest.metadata["embedding_count"] == 2
    assert manifest.metadata["unique_chunk_count"] == 2
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 3


def test_run_embedding_runner_metadata_cannot_override_system_fields(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.metadata = {
        "embedding_count": 999,
        "unique_chunk_count": 999,
        "unique_doc_count": 999,
        "embedding_dim": 999,
        "provenance": {"model_name": "wrong"},
        "stage": "embedding",
    }

    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert manifest.metadata["embedding_count"] == 2
    assert manifest.metadata["unique_chunk_count"] == 2
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 3
    assert manifest.metadata["stage"] == "embedding"
    assert manifest.metadata["provenance"]["model_name"] == "fake-embedding-model"


def test_run_embedding_raises_for_wrong_vector_count(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    with pytest.raises(EmbeddingRunError, match="different number of vectors"):
        run_embedding(source_store, output_store, _config(), WrongCountEmbeddingClient())
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_for_wrong_vector_dimension(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    with pytest.raises(EmbeddingRunError, match="unexpected dimension"):
        run_embedding(source_store, output_store, _config(), WrongDimEmbeddingClient())
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_when_consistency_check_failed(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]
    config.consistency_check = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        passed=False,
        failure_reason="endpoint mismatch",
        max_abs_diff=0.1,
    )

    with pytest.raises(EmbeddingRunError, match="consistency check failed"):
        run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_sharded_source_writes_aligned_embedding_shards(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    write_chunked_corpus_artifact(
        source_store,
        "litsearch_chunks_sharded",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(chunk_id="chunk-1", doc_id="doc-1", text="a", chunk_index=0),
                ChunkRecord(chunk_id="chunk-2", doc_id="doc-1", text="b", chunk_index=1),
                ChunkRecord(chunk_id="chunk-3", doc_id="doc-2", text="c", chunk_index=0),
            ],
            metadata={"source": "chunking"},
        ),
        file_record_num=1,
    )
    config = EmbeddingRunConfig(
        source_artifact_id="litsearch_chunks_sharded",
        output_artifact_id="litsearch_embeddings_sharded",
        model_name="fake-embedding-model",
        embedding_dim=3,
        provider="fake-provider",
    )

    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))
    loaded = read_embeddings_artifact(output_store, "litsearch_embeddings_sharded")
    shards = iter_embedding_shards(output_store, "litsearch_embeddings_sharded")
    source_shards = iter_chunk_shards(source_store, "litsearch_chunks_sharded")

    assert manifest.metadata["sharding"]["enabled"] is True
    assert len(manifest.metadata["shards"]) == 2
    assert [record.chunk_id for record in loaded.embeddings] == ["chunk-1", "chunk-2", "chunk-3"]
    for source_shard, embedding_shard in zip(source_shards, shards, strict=True):
        assert source_shard.shard_id == embedding_shard.shard_id
        assert [chunk.chunk_id for chunk in source_shard.chunks] == [
            record.chunk_id for record in embedding_shard.embeddings
        ]
        assert [chunk.doc_id for chunk in source_shard.chunks] == [
            record.doc_id for record in embedding_shard.embeddings
        ]


def test_run_embedding_reuses_valid_existing_shard_without_calling_client(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    _write_resume_source(source_store)
    _put_existing_embedding_shard(
        output_store,
        "part-00000",
        [
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2, 0.3]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.4, 0.5, 0.6]),
        ],
    )
    client = RecordingEmbeddingClient()
    events: list[ProgressEvent] = []

    manifest = run_embedding(
        source_store,
        output_store,
        _resume_config(),
        client,
        progress_reporter=events.append,
    )
    loaded = read_embeddings_artifact(output_store, "resume_embeddings")

    assert client.calls == [["third chunk text"]]
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "resume_embeddings") is True
    assert [file.path for file in manifest.files] == [
        "embeddings/part-00000.jsonl",
        "embeddings/part-00001.jsonl",
    ]
    assert [shard["shard_id"] for shard in manifest.metadata["shards"]] == [
        "part-00000",
        "part-00001",
    ]
    assert manifest.metadata["embedding_count"] == 3
    assert manifest.metadata["unique_chunk_count"] == 3
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["resume_existing_shards"] is True
    assert manifest.metadata["resumed_shard_count"] == 1
    assert manifest.metadata["computed_shard_count"] == 1
    assert [record.chunk_id for record in loaded.embeddings] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [event.metadata.get("kind") for event in events] == [
        "resume_shard",
        "batch",
        "shard",
    ]


def test_run_embedding_rejects_resumed_shard_with_chunk_id_order_mismatch(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    _write_resume_source(source_store)
    _put_existing_embedding_shard(
        output_store,
        "part-00000",
        [
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.1, 0.2, 0.3]),
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.4, 0.5, 0.6]),
        ],
    )

    with pytest.raises(
        EmbeddingRunError,
        match=r"part-00000: chunk_id order mismatch",
    ):
        run_embedding(source_store, output_store, _resume_config(), RecordingEmbeddingClient())

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "resume_embeddings") is False


def test_run_embedding_rejects_resumed_shard_with_doc_id_mismatch(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    _write_resume_source(source_store)
    _put_existing_embedding_shard(
        output_store,
        "part-00000",
        [
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-wrong", vector=[0.1, 0.2, 0.3]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.4, 0.5, 0.6]),
        ],
    )

    with pytest.raises(
        EmbeddingRunError,
        match=r"part-00000: doc_id order mismatch",
    ):
        run_embedding(source_store, output_store, _resume_config(), RecordingEmbeddingClient())

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "resume_embeddings") is False


def test_run_embedding_rejects_resumed_shard_with_vector_dim_mismatch(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    _write_resume_source(source_store)
    _put_existing_embedding_shard(
        output_store,
        "part-00000",
        [
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.4, 0.5, 0.6]),
        ],
    )

    with pytest.raises(
        EmbeddingRunError,
        match=r"part-00000: vector dimension mismatch",
    ):
        run_embedding(source_store, output_store, _resume_config(), RecordingEmbeddingClient())

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "resume_embeddings") is False


def test_run_embedding_resume_existing_shards_false_recomputes_existing_shard(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    _write_resume_source(source_store)
    _put_existing_embedding_shard(
        output_store,
        "part-00000",
        [
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2, 0.3]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.4, 0.5, 0.6]),
        ],
    )
    client = RecordingEmbeddingClient()

    manifest = run_embedding(
        source_store,
        output_store,
        _resume_config(resume_existing_shards=False),
        client,
    )

    assert client.calls == [["first chunk text", "second chunk text"], ["third chunk text"]]
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "resume_embeddings") is True
    assert manifest.metadata["resume_existing_shards"] is False
    assert manifest.metadata["resumed_shard_count"] == 0
    assert manifest.metadata["computed_shard_count"] == 2


def test_run_embedding_reports_batch_and_shard_progress(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    events: list[ProgressEvent] = []
    config = _config()
    config.batch_size = 1

    run_embedding(
        source_store,
        output_store,
        config,
        FakeEmbeddingClient(3),
        progress_reporter=events.append,
    )

    batch_events = [event for event in events if event.metadata.get("kind") == "batch"]
    shard_events = [event for event in events if event.metadata.get("kind") == "shard"]
    assert len(batch_events) == 2
    assert batch_events[0].metadata["shard_id"] == "part-00000"
    assert batch_events[-1].current == 2
    assert len(shard_events) == 1
    assert shard_events[0].metadata["shard_id"] == "part-00000"


def test_run_embedding_sharded_source_does_not_touch_legacy_single_file_path(
    tmp_path: Path,
    output_store: LocalArtifactStore,
) -> None:
    class RejectLegacySingleFileStore(LocalArtifactStore):
        def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
            if relative_path == "chunks.jsonl":
                raise AssertionError("legacy single-file chunk path should not be read")
            return super().get_file(artifact_type, artifact_id, relative_path)

    source_store = RejectLegacySingleFileStore(tmp_path / "source-sharded")
    write_chunked_corpus_artifact(
        source_store,
        "litsearch_chunks_sharded",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(chunk_id="chunk-1", doc_id="doc-1", text="a", chunk_index=0),
                ChunkRecord(chunk_id="chunk-2", doc_id="doc-2", text="b", chunk_index=0),
            ]
        ),
        file_record_num=1,
    )
    config = EmbeddingRunConfig(
        source_artifact_id="litsearch_chunks_sharded",
        output_artifact_id="litsearch_embeddings_sharded",
        model_name="fake-embedding-model",
        embedding_dim=3,
    )

    run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert output_store.is_complete("embeddings", "litsearch_embeddings_sharded") is True


def test_run_embedding_does_not_call_full_chunked_corpus_reader(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("full chunked corpus reader should not be used")

    monkeypatch.setattr(runner_module, "read_chunked_corpus_artifact", fail_read, raising=False)

    run_embedding(source_store, output_store, _config(), FakeEmbeddingClient(3))

    assert output_store.is_complete("embeddings", "litsearch_embeddings") is True


def test_run_embedding_progress_reporter_failure_does_not_write_success(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.batch_size = 1

    def failing_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_embedding(
            source_store,
            output_store,
            config,
            FakeEmbeddingClient(3),
            progress_reporter=failing_reporter,
        )

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_when_multi_endpoint_check_missing(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]

    with pytest.raises(EmbeddingRunError, match="require a consistency_check result"):
        run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_allows_single_endpoint_without_consistency_check(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_id = "endpoint-a"
    config.endpoint_ids = ["endpoint-a"]

    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert manifest.artifact_id == "litsearch_embeddings"
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is True
