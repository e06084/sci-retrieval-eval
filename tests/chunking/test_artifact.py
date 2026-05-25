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
    load_chunks_jsonl,
    read_chunked_corpus_artifact,
    write_chunked_corpus_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


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
