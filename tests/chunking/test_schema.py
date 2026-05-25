"""Tests for chunking schema models."""

import pytest
from pydantic import ValidationError

from eval_platform.chunking import ChunkedCorpus, ChunkerProvenance, ChunkRecord


def test_chunk_record_construction() -> None:
    record = ChunkRecord(
        chunk_id="c-1",
        doc_id="doc-1",
        text="first chunk",
        title="Example",
        chunk_index=0,
        start_offset=0,
        end_offset=12,
        metadata={"section": "intro"},
    )

    assert record.chunk_id == "c-1"
    assert record.doc_id == "doc-1"
    assert record.text == "first chunk"
    assert record.chunk_index == 0
    assert record.start_offset == 0
    assert record.end_offset == 12
    assert record.metadata == {"section": "intro"}


def test_chunked_corpus_construction() -> None:
    corpus = ChunkedCorpus(
        chunks=[
            ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first", chunk_index=0),
            ChunkRecord(chunk_id="c-2", doc_id="doc-1", text="second", chunk_index=1),
        ],
        metadata={"source_normalized_dataset_artifact_id": "litsearch_test"},
    )

    assert len(corpus.chunks) == 2
    assert corpus.metadata == {"source_normalized_dataset_artifact_id": "litsearch_test"}


def test_chunk_record_metadata_defaults_are_independent() -> None:
    first = ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first", chunk_index=0)
    second = ChunkRecord(chunk_id="c-2", doc_id="doc-2", text="second", chunk_index=0)

    first.metadata["key"] = "a"
    second.metadata["key"] = "b"

    assert first.metadata == {"key": "a"}
    assert second.metadata == {"key": "b"}


def test_chunked_corpus_metadata_defaults_are_independent() -> None:
    first = ChunkedCorpus(
        chunks=[ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first", chunk_index=0)]
    )
    second = ChunkedCorpus(
        chunks=[ChunkRecord(chunk_id="c-2", doc_id="doc-2", text="second", chunk_index=0)]
    )

    first.metadata["key"] = "a"
    second.metadata["key"] = "b"

    assert first.metadata == {"key": "a"}
    assert second.metadata == {"key": "b"}


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunk_id", ""),
        ("doc_id", ""),
        ("text", ""),
        ("chunk_id", " "),
        ("doc_id", " "),
        ("text", " "),
    ],
)
def test_chunk_record_rejects_empty_or_blank_strings(field_name: str, value: str) -> None:
    payload = {
        "chunk_id": "c-1",
        "doc_id": "doc-1",
        "text": "hello",
        "chunk_index": 0,
        field_name: value,
    }

    with pytest.raises(ValidationError):
        ChunkRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunk_index", -1),
        ("start_offset", -1),
        ("end_offset", -1),
    ],
)
def test_chunk_record_rejects_negative_numeric_fields(field_name: str, value: int) -> None:
    payload = {
        "chunk_id": "c-1",
        "doc_id": "doc-1",
        "text": "hello",
        "chunk_index": 0,
        field_name: value,
    }

    with pytest.raises(ValidationError):
        ChunkRecord.model_validate(payload)


def test_chunk_record_rejects_end_offset_before_start_offset() -> None:
    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id="c-1",
            doc_id="doc-1",
            text="hello",
            chunk_index=0,
            start_offset=10,
            end_offset=5,
        )


def test_chunk_record_accepts_equal_offsets() -> None:
    record = ChunkRecord(
        chunk_id="c-1",
        doc_id="doc-1",
        text="hello",
        chunk_index=0,
        start_offset=10,
        end_offset=10,
    )

    assert record.start_offset == 10
    assert record.end_offset == 10


def test_chunk_record_accepts_end_offset_after_start_offset() -> None:
    record = ChunkRecord(
        chunk_id="c-1",
        doc_id="doc-1",
        text="hello",
        chunk_index=0,
        start_offset=10,
        end_offset=20,
    )

    assert record.start_offset == 10
    assert record.end_offset == 20


def test_chunker_provenance_construction() -> None:
    provenance = ChunkerProvenance(
        name="sciverse-chunker",
        repo_url="https://example.com/sciverse-chunker.git",
        repo_path="/tmp/sciverse-chunker",
        commit_sha="abc123def456",
        branch="main",
        is_dirty=True,
        metadata={"runtime": "test"},
    )

    assert provenance.name == "sciverse-chunker"
    assert provenance.commit_sha == "abc123def456"
    assert provenance.is_dirty is True
    assert provenance.metadata == {"runtime": "test"}


@pytest.mark.parametrize("name", ["", " "])
def test_chunker_provenance_rejects_empty_name(name: str) -> None:
    with pytest.raises(ValidationError):
        ChunkerProvenance(name=name, commit_sha="abc123")


@pytest.mark.parametrize("commit_sha", ["", " "])
def test_chunker_provenance_rejects_empty_commit_sha(commit_sha: str) -> None:
    with pytest.raises(ValidationError):
        ChunkerProvenance(name="sciverse-chunker", commit_sha=commit_sha)


def test_chunker_provenance_is_dirty_defaults_to_false() -> None:
    provenance = ChunkerProvenance(name="sciverse-chunker", commit_sha="abc123")

    assert provenance.is_dirty is False


def test_chunker_provenance_metadata_defaults_are_independent() -> None:
    first = ChunkerProvenance(name="chunker-a", commit_sha="abc123")
    second = ChunkerProvenance(name="chunker-b", commit_sha="def456")

    first.metadata["key"] = "a"
    second.metadata["key"] = "b"

    assert first.metadata == {"key": "a"}
    assert second.metadata == {"key": "b"}
