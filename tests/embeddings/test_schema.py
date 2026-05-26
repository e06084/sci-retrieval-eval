"""Tests for embedding schemas."""

import math

import pytest
from pydantic import ValidationError

from eval_platform.embeddings import EmbeddedCorpus, EmbeddingProvenance, EmbeddingRecord


def test_embedding_provenance_constructs() -> None:
    provenance = EmbeddingProvenance(
        model_name="text-embedding-3-large",
        provider="fake",
        api_version="2026-01-01",
        embedding_dim=3072,
        normalized=True,
    )

    assert provenance.model_name == "text-embedding-3-large"
    assert provenance.embedding_dim == 3072


@pytest.mark.parametrize("value", ["", " "])
def test_embedding_provenance_rejects_blank_model_name(value: str) -> None:
    with pytest.raises(ValidationError):
        EmbeddingProvenance(model_name=value, embedding_dim=3)


@pytest.mark.parametrize("value", [0, -1])
def test_embedding_provenance_rejects_non_positive_dimension(value: int) -> None:
    with pytest.raises(ValidationError):
        EmbeddingProvenance(model_name="model", embedding_dim=value)


def test_embedding_provenance_default_metadata_not_shared() -> None:
    first = EmbeddingProvenance(model_name="a", embedding_dim=3)
    second = EmbeddingProvenance(model_name="b", embedding_dim=3)
    first.metadata["x"] = 1
    second.metadata["x"] = 2
    assert first.metadata == {"x": 1}
    assert second.metadata == {"x": 2}


def test_embedding_record_constructs() -> None:
    record = EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2])
    assert record.vector == [0.1, 0.2]


@pytest.mark.parametrize("value", ["", " "])
def test_embedding_record_rejects_blank_chunk_id(value: str) -> None:
    with pytest.raises(ValidationError):
        EmbeddingRecord(chunk_id=value, doc_id="doc-1", vector=[0.1])


@pytest.mark.parametrize("value", ["", " "])
def test_embedding_record_rejects_blank_doc_id(value: str) -> None:
    with pytest.raises(ValidationError):
        EmbeddingRecord(chunk_id="chunk-1", doc_id=value, vector=[0.1])


def test_embedding_record_rejects_empty_vector() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_embedding_record_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, value])


def test_embedding_record_default_metadata_not_shared() -> None:
    first = EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1])
    second = EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-2", vector=[0.2])
    first.metadata["a"] = 1
    second.metadata["a"] = 2
    assert first.metadata == {"a": 1}
    assert second.metadata == {"a": 2}


def test_embedded_corpus_constructs() -> None:
    corpus = EmbeddedCorpus(
        embeddings=[EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1])]
    )
    assert len(corpus.embeddings) == 1


def test_embedded_corpus_default_metadata_not_shared() -> None:
    first = EmbeddedCorpus()
    second = EmbeddedCorpus()
    first.metadata["a"] = 1
    second.metadata["a"] = 2
    assert first.metadata == {"a": 1}
    assert second.metadata == {"a": 2}
