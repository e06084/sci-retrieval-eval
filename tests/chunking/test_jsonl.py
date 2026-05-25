"""Tests for chunk JSONL helpers."""

import json

import pytest
from pydantic import ValidationError

from eval_platform.chunking import ChunkRecord, dump_chunks_jsonl, load_chunks_jsonl


def test_dump_chunks_jsonl_one_record_per_line() -> None:
    chunks = [
        ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first", chunk_index=0),
        ChunkRecord(chunk_id="c-2", doc_id="doc-2", text="second", chunk_index=0),
    ]

    text = dump_chunks_jsonl(chunks)
    lines = text.splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["chunk_id"] == "c-1"
    assert json.loads(lines[1])["chunk_id"] == "c-2"


def test_dump_chunks_jsonl_ends_with_newline() -> None:
    text = dump_chunks_jsonl(
        [ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="hello", chunk_index=0)]
    )

    assert text.endswith("\n")


def test_dump_chunks_jsonl_empty_input() -> None:
    assert dump_chunks_jsonl([]) == ""


def test_load_chunks_jsonl_round_trip() -> None:
    chunks = [
        ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="first", chunk_index=0),
        ChunkRecord(chunk_id="c-2", doc_id="doc-2", text="second", chunk_index=1),
    ]

    loaded = load_chunks_jsonl(dump_chunks_jsonl(chunks))

    assert loaded == chunks


def test_load_chunks_jsonl_ignores_blank_lines() -> None:
    text = '\n{"chunk_id":"c-1","doc_id":"doc-1","text":"hello","chunk_index":0}\n\n'

    loaded = load_chunks_jsonl(text)

    assert loaded == [ChunkRecord(chunk_id="c-1", doc_id="doc-1", text="hello", chunk_index=0)]


def test_load_chunks_jsonl_invalid_json_raises() -> None:
    with pytest.raises(ValidationError):
        load_chunks_jsonl("{not-json")


def test_load_chunks_jsonl_invalid_schema_raises() -> None:
    with pytest.raises(ValidationError):
        load_chunks_jsonl('{"chunk_id":"","doc_id":"doc-1","text":"hello","chunk_index":0}')


def test_load_chunks_jsonl_missing_chunk_index_raises() -> None:
    with pytest.raises(ValidationError):
        load_chunks_jsonl('{"chunk_id":"c-1","doc_id":"doc-1","text":"hello"}')
