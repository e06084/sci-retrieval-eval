"""Tests for dataset JSONL helpers."""

import json

import pytest
from pydantic import ValidationError

from eval_platform.datasets import CorpusRecord, QueryRecord, dump_jsonl, load_jsonl


def test_dump_jsonl_one_record_per_line() -> None:
    records = [
        CorpusRecord(doc_id="doc-1", text="first"),
        CorpusRecord(doc_id="doc-2", text="second"),
    ]

    text = dump_jsonl(records)
    lines = text.splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["doc_id"] == "doc-1"
    assert json.loads(lines[1])["doc_id"] == "doc-2"


def test_dump_jsonl_ends_with_newline() -> None:
    text = dump_jsonl([CorpusRecord(doc_id="doc-1", text="hello")])

    assert text.endswith("\n")


def test_dump_jsonl_empty_input() -> None:
    assert dump_jsonl([]) == ""


def test_load_jsonl_round_trip() -> None:
    records = [
        QueryRecord(query_id="q-1", text="first"),
        QueryRecord(query_id="q-2", text="second"),
    ]

    loaded = load_jsonl(dump_jsonl(records), QueryRecord)

    assert loaded == records


def test_load_jsonl_ignores_blank_lines() -> None:
    text = '\n{"query_id":"q-1","text":"hello"}\n\n'

    loaded = load_jsonl(text, QueryRecord)

    assert loaded == [QueryRecord(query_id="q-1", text="hello")]


def test_load_jsonl_invalid_json_raises() -> None:
    with pytest.raises(ValidationError):
        load_jsonl("{not-json", QueryRecord)


def test_load_jsonl_invalid_schema_raises() -> None:
    with pytest.raises(ValidationError):
        load_jsonl('{"query_id":"","text":"hello"}', QueryRecord)
