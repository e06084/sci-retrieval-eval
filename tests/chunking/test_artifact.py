"""Tests for chunked corpus artifact helpers."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    LocalArtifactStore,
)
from eval_platform.chunking import (
    CHUNKS_FILENAME,
    ChunkedCorpus,
    ChunkerProvenance,
    ChunkRecord,
    dump_chunks_jsonl,
    iter_chunk_shards,
    load_chunks_jsonl,
    read_chunked_corpus_artifact,
    write_chunked_corpus_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


class CountingLocalArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.get_file_calls: list[str] = []

    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        self.get_file_calls.append(relative_path)
        return super().get_file(artifact_type, artifact_id, relative_path)


def _sample_corpus() -> ChunkedCorpus:
    return ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first chunk", chunk_index=0),
            ChunkRecord(chunk_id="c-2", doc_id="doc-1", text="second chunk", chunk_index=1),
            ChunkRecord(chunk_id="c-3", doc_id="doc-2", text="other doc", chunk_index=0),
        ],
        metadata={"source_normalized_dataset_artifact_id": "litsearch_test"},
    )


def test_write_chunked_corpus_artifact_writes_chunks_jsonl(store: LocalArtifactStore) -> None:
    corpus = _sample_corpus()

    manifest = write_chunked_corpus_artifact(store, "litsearch_test", corpus)

    chunks_bytes = store.get_file("chunked_corpus", "litsearch_test", CHUNKS_FILENAME)
    assert load_chunks_jsonl(chunks_bytes.decode("utf-8")) == corpus.chunks
    assert any(file.path == CHUNKS_FILENAME for file in manifest.files)
    assert manifest.files[0].sha256 is not None


def test_read_chunked_corpus_artifact_round_trip(store: LocalArtifactStore) -> None:
    corpus = _sample_corpus()

    write_chunked_corpus_artifact(store, "litsearch_test", corpus)
    loaded = read_chunked_corpus_artifact(store, "litsearch_test")

    assert loaded.chunks == corpus.chunks
    assert loaded.metadata == corpus.metadata


def test_write_chunked_corpus_artifact_manifest_lists_chunks_jsonl(
    store: LocalArtifactStore,
) -> None:
    manifest = write_chunked_corpus_artifact(store, "litsearch_test", _sample_corpus())

    assert [file.path for file in manifest.files] == [CHUNKS_FILENAME]


def test_write_chunked_corpus_artifact_shards_by_source_doc_count(
    store: LocalArtifactStore,
) -> None:
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="doc-1-0", doc_id="doc-1", text="a", chunk_index=0),
            ChunkRecord(chunk_id="doc-1-1", doc_id="doc-1", text="b", chunk_index=1),
            ChunkRecord(chunk_id="doc-2-0", doc_id="doc-2", text="c", chunk_index=0),
            ChunkRecord(chunk_id="doc-3-0", doc_id="doc-3", text="d", chunk_index=0),
            ChunkRecord(chunk_id="doc-3-1", doc_id="doc-3", text="e", chunk_index=1),
        ]
    )

    manifest = write_chunked_corpus_artifact(
        store,
        "sharded_chunks",
        corpus,
        file_record_num=2,
    )

    assert store.exists("chunked_corpus", "sharded_chunks", "chunks/part-00000.jsonl")
    assert store.exists("chunked_corpus", "sharded_chunks", "chunks/part-00001.jsonl")
    assert not store.exists("chunked_corpus", "sharded_chunks", CHUNKS_FILENAME)
    assert manifest.metadata["sharding"]["enabled"] is True
    assert manifest.metadata["sharding"]["file_record_num"] == 2
    assert len(manifest.metadata["shards"]) == 2
    assert manifest.metadata["shards"][0]["source_doc_count"] == 2
    assert manifest.metadata["shards"][0]["chunk_count"] == 3
    assert manifest.metadata["shards"][1]["source_doc_count"] == 1
    assert manifest.metadata["shards"][1]["chunk_count"] == 2


def test_read_chunked_corpus_artifact_reads_sharded_artifact_in_order(
    store: LocalArtifactStore,
) -> None:
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="doc-1-0", doc_id="doc-1", text="a", chunk_index=0),
            ChunkRecord(chunk_id="doc-1-1", doc_id="doc-1", text="b", chunk_index=1),
            ChunkRecord(chunk_id="doc-2-0", doc_id="doc-2", text="c", chunk_index=0),
            ChunkRecord(chunk_id="doc-3-0", doc_id="doc-3", text="d", chunk_index=0),
        ],
        metadata={"source": "unit-test"},
    )
    write_chunked_corpus_artifact(store, "sharded_chunks", corpus, file_record_num=2)

    loaded = read_chunked_corpus_artifact(store, "sharded_chunks")

    assert [chunk.chunk_id for chunk in loaded.chunks] == [
        chunk.chunk_id for chunk in corpus.chunks
    ]
    assert loaded.metadata == {"source": "unit-test"}


def test_iter_chunk_shards_returns_manifest_order(store: LocalArtifactStore) -> None:
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="doc-1-0", doc_id="doc-1", text="a", chunk_index=0),
            ChunkRecord(chunk_id="doc-2-0", doc_id="doc-2", text="b", chunk_index=0),
            ChunkRecord(chunk_id="doc-3-0", doc_id="doc-3", text="c", chunk_index=0),
        ]
    )
    write_chunked_corpus_artifact(store, "sharded_chunks", corpus, file_record_num=2)

    shards = list(iter_chunk_shards(store, "sharded_chunks"))

    assert [shard.shard_id for shard in shards] == ["part-00000", "part-00001"]
    assert [chunk.chunk_id for chunk in shards[0].chunks] == ["doc-1-0", "doc-2-0"]
    assert [chunk.chunk_id for chunk in shards[1].chunks] == ["doc-3-0"]


def test_iter_chunk_shards_is_lazy(tmp_path: Path) -> None:
    store = CountingLocalArtifactStore(tmp_path)
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="doc-1-0", doc_id="doc-1", text="a", chunk_index=0),
            ChunkRecord(chunk_id="doc-2-0", doc_id="doc-2", text="b", chunk_index=0),
            ChunkRecord(chunk_id="doc-3-0", doc_id="doc-3", text="c", chunk_index=0),
        ]
    )
    write_chunked_corpus_artifact(store, "sharded_chunks", corpus, file_record_num=2)

    shard_iter = iter_chunk_shards(store, "sharded_chunks")

    assert not isinstance(shard_iter, list)
    assert store.get_file_calls == []

    first_shard = next(shard_iter)
    assert first_shard.shard_id == "part-00000"
    assert store.get_file_calls == ["chunks/part-00000.jsonl"]

    second_shard = next(shard_iter)
    assert second_shard.shard_id == "part-00001"
    assert store.get_file_calls == ["chunks/part-00000.jsonl", "chunks/part-00001.jsonl"]


def test_shard_file_sha256_matches_file_payload(store: LocalArtifactStore) -> None:
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="doc-1-0", doc_id="doc-1", text="a", chunk_index=0),
            ChunkRecord(chunk_id="doc-2-0", doc_id="doc-2", text="b", chunk_index=0),
        ]
    )
    manifest = write_chunked_corpus_artifact(store, "sharded_chunks", corpus, file_record_num=1)

    shard_file = manifest.files[0]
    payload = store.get_file("chunked_corpus", "sharded_chunks", shard_file.path)

    import hashlib

    assert hashlib.sha256(payload).hexdigest() == shard_file.sha256
    assert manifest.metadata["shards"][0]["sha256"] == shard_file.sha256


def test_write_chunked_corpus_artifact_rejects_non_positive_file_record_num(
    store: LocalArtifactStore,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        write_chunked_corpus_artifact(store, "bad_chunks", _sample_corpus(), file_record_num=0)


def test_write_chunked_corpus_artifact_empty_sharded_corpus_writes_zero_shards(
    store: LocalArtifactStore,
) -> None:
    manifest = write_chunked_corpus_artifact(
        store,
        "empty_chunks",
        ChunkedCorpus(chunks=[]),
        file_record_num=2,
    )

    assert manifest.metadata["chunk_count"] == 0
    assert manifest.metadata["unique_doc_count"] == 0
    assert manifest.metadata["shards"] == []
    assert manifest.files == []
    assert read_chunked_corpus_artifact(store, "empty_chunks").chunks == []


def test_read_chunked_corpus_artifact_requires_success_marker(store: LocalArtifactStore) -> None:
    corpus = _sample_corpus()
    artifact_id = "incomplete_chunks"
    chunks_bytes = dump_chunks_jsonl(corpus.chunks).encode("utf-8")

    store.put_file("chunked_corpus", artifact_id, CHUNKS_FILENAME, chunks_bytes)
    store.write_manifest(
        "chunked_corpus",
        artifact_id,
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="chunked_corpus",
            created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
            files=[ArtifactFile(path=CHUNKS_FILENAME, size_bytes=len(chunks_bytes))],
        ),
    )

    with pytest.raises(ArtifactIncompleteError):
        read_chunked_corpus_artifact(store, artifact_id)


def test_read_chunked_corpus_artifact_strips_system_metadata(store: LocalArtifactStore) -> None:
    chunker = ChunkerProvenance(name="sciverse-chunker", commit_sha="deadbeef123456")

    write_chunked_corpus_artifact(
        store,
        "litsearch_test",
        _sample_corpus(),
        chunker=chunker,
        chunk_params={"max_tokens": 512},
        metadata={
            "chunk_count": 999,
            "unique_doc_count": 888,
            "pipeline_step": "chunk",
        },
    )

    loaded = read_chunked_corpus_artifact(store, "litsearch_test")

    assert "chunk_count" not in loaded.metadata
    assert "unique_doc_count" not in loaded.metadata
    assert "chunker" not in loaded.metadata
    assert "chunk_params" not in loaded.metadata
    assert "sharding" not in loaded.metadata
    assert "shards" not in loaded.metadata
    assert loaded.metadata["source_normalized_dataset_artifact_id"] == "litsearch_test"
    assert loaded.metadata["pipeline_step"] == "chunk"


def test_write_chunked_corpus_artifact_records_chunker_provenance(
    store: LocalArtifactStore,
) -> None:
    chunker = ChunkerProvenance(
        name="sciverse-chunker",
        repo_url="https://example.com/sciverse-chunker.git",
        commit_sha="deadbeef123456",
        branch="main",
    )

    manifest = write_chunked_corpus_artifact(
        store,
        "litsearch_test",
        _sample_corpus(),
        chunker=chunker,
    )

    assert "chunker" in manifest.metadata
    assert manifest.metadata["chunker"]["commit_sha"] == "deadbeef123456"
    assert manifest.metadata["chunker"]["name"] == "sciverse-chunker"


def test_write_chunked_corpus_artifact_records_chunk_params(store: LocalArtifactStore) -> None:
    manifest = write_chunked_corpus_artifact(
        store,
        "litsearch_test",
        _sample_corpus(),
        chunk_params={"max_tokens": 512, "overlap": 64},
    )

    assert manifest.metadata["chunk_params"] == {"max_tokens": 512, "overlap": 64}


def test_manifest_count_metadata_is_not_overridden_by_user_metadata(
    store: LocalArtifactStore,
) -> None:
    manifest = write_chunked_corpus_artifact(
        store,
        "litsearch_test",
        _sample_corpus(),
        metadata={
            "chunk_count": 999,
            "unique_doc_count": 888,
            "pipeline_step": "chunk",
        },
    )

    assert manifest.metadata["chunk_count"] == 3
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["pipeline_step"] == "chunk"


def test_write_chunked_corpus_artifact_records_source_dependency(
    store: LocalArtifactStore,
) -> None:
    source = ArtifactDependency(
        artifact_id="litsearch_test",
        artifact_type="normalized_dataset",
    )

    manifest = write_chunked_corpus_artifact(
        store,
        "litsearch_test_chunks",
        _sample_corpus(),
        source_dependency=source,
    )

    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].artifact_id == "litsearch_test"
    assert manifest.dependencies[0].artifact_type == "normalized_dataset"

    persisted = store.read_manifest("chunked_corpus", "litsearch_test_chunks")
    assert persisted.dependencies == manifest.dependencies


def test_write_chunked_corpus_artifact_marks_complete(store: LocalArtifactStore) -> None:
    manifest = write_chunked_corpus_artifact(store, "litsearch_test", _sample_corpus())

    assert store.is_complete("chunked_corpus", manifest.artifact_id) is True
