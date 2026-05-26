"""Tests for embeddings JSONL helpers."""

import json

import pytest
from pydantic import ValidationError

from eval_platform.embeddings import EmbeddingRecord, dump_embeddings_jsonl, load_embeddings_jsonl


def _records() -> list[EmbeddingRecord]:
    return [
        EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
        EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-2", vector=[0.3, 0.4]),
    ]


def test_dump_embeddings_jsonl_writes_one_json_per_line() -> None:
    dumped = dump_embeddings_jsonl(_records())
    lines = dumped.strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["chunk_id"] == "chunk-1"
    assert json.loads(lines[1])["chunk_id"] == "chunk-2"


def test_dump_embeddings_jsonl_ends_with_newline() -> None:
    assert dump_embeddings_jsonl(_records()).endswith("\n")


def test_dump_embeddings_jsonl_empty_input_returns_empty_string() -> None:
    assert dump_embeddings_jsonl([]) == ""


def test_load_embeddings_jsonl_round_trip() -> None:
    records = _records()
    assert load_embeddings_jsonl(dump_embeddings_jsonl(records)) == records


def test_load_embeddings_jsonl_ignores_empty_lines() -> None:
    text = dump_embeddings_jsonl(_records()) + "\n\n"
    assert len(load_embeddings_jsonl(text)) == 2


def test_load_embeddings_jsonl_raises_for_invalid_json() -> None:
    with pytest.raises(ValidationError, match="Invalid JSON"):
        load_embeddings_jsonl('{"chunk_id":"chunk-1"\n')


def test_load_embeddings_jsonl_raises_for_invalid_schema() -> None:
    with pytest.raises(ValidationError):
        load_embeddings_jsonl('{"chunk_id":"chunk-1","doc_id":"doc-1","vector":[]}\n')
