"""Tests for dataset schema models."""

from typing import TypeAlias

import pytest
from pydantic import ValidationError

from eval_platform.datasets import CorpusRecord, NormalizedDataset, QrelRecord, QueryRecord

DatasetRecordModel: TypeAlias = type[CorpusRecord] | type[QueryRecord] | type[QrelRecord]


def test_corpus_record_construction() -> None:
    record = CorpusRecord(doc_id="doc-1", text="hello", title="Title")

    assert record.doc_id == "doc-1"
    assert record.text == "hello"
    assert record.title == "Title"
    assert record.metadata == {}


def test_query_record_construction() -> None:
    record = QueryRecord(query_id="q-1", text="what is science?")

    assert record.query_id == "q-1"
    assert record.text == "what is science?"
    assert record.metadata == {}


def test_qrel_record_construction() -> None:
    record = QrelRecord(query_id="q-1", doc_id="doc-1", relevance=2.0)

    assert record.query_id == "q-1"
    assert record.doc_id == "doc-1"
    assert record.relevance == 2.0
    assert record.metadata == {}


def test_metadata_defaults_are_independent() -> None:
    corpus_a = CorpusRecord(doc_id="doc-a", text="a")
    corpus_b = CorpusRecord(doc_id="doc-b", text="b")
    query = QueryRecord(query_id="q-1", text="query")
    qrel = QrelRecord(query_id="q-1", doc_id="doc-a")
    dataset = NormalizedDataset(corpus=[corpus_a], queries=[query], qrels=[qrel])

    corpus_a.metadata["source"] = "a"
    corpus_b.metadata["source"] = "b"
    query.metadata["split"] = "test"
    qrel.metadata["judge"] = "human"
    dataset.metadata["dataset"] = "sample"

    assert corpus_a.metadata == {"source": "a"}
    assert corpus_b.metadata == {"source": "b"}
    assert query.metadata == {"split": "test"}
    assert qrel.metadata == {"judge": "human"}
    assert dataset.metadata == {"dataset": "sample"}


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (CorpusRecord, {"doc_id": "", "text": "hello"}),
        (CorpusRecord, {"doc_id": "doc-1", "text": ""}),
        (QueryRecord, {"query_id": "", "text": "query"}),
        (QueryRecord, {"query_id": "q-1", "text": ""}),
        (QrelRecord, {"query_id": "", "doc_id": "doc-1"}),
        (QrelRecord, {"query_id": "q-1", "doc_id": ""}),
    ],
)
def test_empty_identifiers_or_text_are_rejected(
    factory: DatasetRecordModel, payload: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        factory.model_validate(payload)


def test_qrel_rejects_negative_relevance() -> None:
    with pytest.raises(ValidationError):
        QrelRecord(query_id="q-1", doc_id="doc-1", relevance=-0.1)


def test_qrel_supports_graded_relevance() -> None:
    record = QrelRecord(query_id="q-1", doc_id="doc-1", relevance=2.0)

    assert record.relevance == 2.0


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (CorpusRecord, {"doc_id": " ", "text": "hello"}),
        (CorpusRecord, {"doc_id": "doc-1", "text": " "}),
        (QueryRecord, {"query_id": " ", "text": "query"}),
        (QueryRecord, {"query_id": "q-1", "text": " "}),
        (QrelRecord, {"query_id": " ", "doc_id": "doc-1"}),
        (QrelRecord, {"query_id": "q-1", "doc_id": " "}),
    ],
)
def test_whitespace_only_identifiers_or_text_are_rejected(
    factory: DatasetRecordModel, payload: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        factory.model_validate(payload)
