"""Tests for embeddings JSONL helpers."""

import json

import pytest

from eval_platform.embeddings import (
    VECTOR_ENCODING,
    EmbeddingRecord,
    dump_embeddings_jsonl,
    load_embeddings_jsonl,
)


def _records() -> list[EmbeddingRecord]:
    return [
        EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
        EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-2", vector=[0.3, 0.4]),
    ]


def test_dump_embeddings_jsonl_writes_one_json_per_line() -> None:
    dumped = dump_embeddings_jsonl(_records())
    lines = dumped.strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["chunk_id"] == "chunk-1"
    assert first["vector_encoding"] == VECTOR_ENCODING
    assert isinstance(first["vector_b64"], str)
    assert "vector" not in first
    assert second["chunk_id"] == "chunk-2"


def test_dump_embeddings_jsonl_ends_with_newline() -> None:
    assert dump_embeddings_jsonl(_records()).endswith("\n")


def test_dump_embeddings_jsonl_empty_input_returns_empty_string() -> None:
    assert dump_embeddings_jsonl([]) == ""


def test_load_embeddings_jsonl_round_trip() -> None:
    records = _records()
    loaded = load_embeddings_jsonl(dump_embeddings_jsonl(records))
    assert [record.chunk_id for record in loaded] == [record.chunk_id for record in records]
    assert [record.doc_id for record in loaded] == [record.doc_id for record in records]
    assert loaded[0].vector == pytest.approx(records[0].vector)
    assert loaded[1].vector == pytest.approx(records[1].vector)


def test_load_embeddings_jsonl_ignores_empty_lines() -> None:
    text = dump_embeddings_jsonl(_records()) + "\n\n"
    assert len(load_embeddings_jsonl(text)) == 2


def test_load_embeddings_jsonl_raises_for_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        load_embeddings_jsonl('{"chunk_id":"chunk-1"\n')


def test_load_embeddings_jsonl_raises_for_invalid_schema() -> None:
    with pytest.raises(ValueError, match="vector_encoding"):
        load_embeddings_jsonl('{"chunk_id":"chunk-1","doc_id":"doc-1","vector":[]}\n')
