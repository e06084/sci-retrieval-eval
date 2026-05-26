"""Tests for embedding schemas."""

import math

import pytest
from pydantic import ValidationError

from eval_platform.embeddings import (
    EmbeddedCorpus,
    EmbeddingConsistencyCheckResult,
    EmbeddingProvenance,
    EmbeddingRecord,
)


def test_embedding_provenance_constructs() -> None:
    provenance = EmbeddingProvenance(
        model_name="text-embedding-3-large",
        provider="fake",
        api_version="2026-01-01",
        embedding_dim=3072,
        normalized=True,
        endpoint_id="endpoint-a",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        consistency_check=EmbeddingConsistencyCheckResult(
            input_text="probe text",
            endpoint_ids=["endpoint-a", "endpoint-b"],
            passed=True,
            max_abs_diff=0.0,
        ),
        runtime_parameters={"batch_size": 8, "timeout_seconds": 60.0},
    )

    assert provenance.model_name == "text-embedding-3-large"
    assert provenance.embedding_dim == 3072
    assert provenance.endpoint_id == "endpoint-a"
    assert provenance.endpoint_ids == ["endpoint-a", "endpoint-b"]
    assert provenance.runtime_parameters["batch_size"] == 8


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


def test_embedding_provenance_default_collections_not_shared() -> None:
    first = EmbeddingProvenance(model_name="a", embedding_dim=3)
    second = EmbeddingProvenance(model_name="b", embedding_dim=3)
    first.endpoint_ids.append("endpoint-a")
    first.runtime_parameters["batch_size"] = 8
    second.endpoint_ids.append("endpoint-b")
    second.runtime_parameters["batch_size"] = 16
    assert first.endpoint_ids == ["endpoint-a"]
    assert second.endpoint_ids == ["endpoint-b"]
    assert first.runtime_parameters == {"batch_size": 8}
    assert second.runtime_parameters == {"batch_size": 16}


@pytest.mark.parametrize("value", ["", " "])
def test_embedding_provenance_rejects_blank_endpoint_id(value: str) -> None:
    with pytest.raises(ValidationError):
        EmbeddingProvenance(model_name="model", embedding_dim=3, endpoint_id=value)


def test_embedding_provenance_rejects_blank_endpoint_ids() -> None:
    with pytest.raises(ValidationError):
        EmbeddingProvenance(model_name="model", embedding_dim=3, endpoint_ids=["endpoint-a", " "])


def test_embedding_consistency_check_result_constructs() -> None:
    result = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        passed=True,
        max_abs_diff=0.0,
    )
    assert result.passed is True
    assert result.endpoint_ids == ["endpoint-a", "endpoint-b"]


def test_embedding_consistency_check_result_rejects_failure_reason_when_passed() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConsistencyCheckResult(
            input_text="probe text",
            endpoint_ids=["endpoint-a"],
            passed=True,
            failure_reason="should not be here",
        )


@pytest.mark.parametrize("value", [None, "", " "])
def test_embedding_consistency_check_result_requires_failure_reason_when_failed(
    value: str | None,
) -> None:
    with pytest.raises(ValidationError):
        EmbeddingConsistencyCheckResult(
            input_text="probe text",
            endpoint_ids=["endpoint-a"],
            passed=False,
            failure_reason=value,
        )


@pytest.mark.parametrize("value", ["", " "])
def test_embedding_consistency_check_result_rejects_blank_input_text(value: str) -> None:
    with pytest.raises(ValidationError):
        EmbeddingConsistencyCheckResult(
            input_text=value,
            endpoint_ids=["endpoint-a"],
            passed=True,
        )


def test_embedding_consistency_check_result_rejects_empty_endpoint_ids() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConsistencyCheckResult(
            input_text="probe text",
            endpoint_ids=[],
            passed=False,
        )


def test_embedding_consistency_check_result_default_metadata_not_shared() -> None:
    first = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a"],
        passed=True,
    )
    second = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-b"],
        passed=True,
    )
    first.metadata["x"] = 1
    second.metadata["x"] = 2
    assert first.metadata == {"x": 1}
    assert second.metadata == {"x": 2}


def test_embedding_consistency_check_result_trims_failure_reason() -> None:
    result = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a"],
        passed=False,
        failure_reason="  failed  ",
    )
    assert result.failure_reason == "failed"


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
