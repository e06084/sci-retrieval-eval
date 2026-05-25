"""Tests for MTEB retrieval data conversion."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from eval_platform.mteb_adapter import (
    MTEBConversionError,
    convert_retrieval_data_to_normalized_dataset,
)


def test_corpus_dict_with_title_and_text() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": {"title": "Title", "text": "Body", "year": 2024}},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.corpus[0].doc_id == "doc-1"
    assert dataset.corpus[0].title == "Title"
    assert dataset.corpus[0].text == "Body"
    assert dataset.corpus[0].metadata == {"year": 2024}


def test_corpus_string_value() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "Plain text"},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.corpus[0].text == "Plain text"
    assert dataset.corpus[0].title is None


def test_corpus_dict_uses_abstract_when_text_missing() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": {"title": "Title", "abstract": "Abstract body"}},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.corpus[0].text == "Abstract body"


def test_corpus_dict_uses_title_when_text_and_abstract_missing() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": {"title": "Title only"}},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.corpus[0].title == "Title only"
    assert dataset.corpus[0].text == "Title only"


def test_corpus_dict_without_text_abstract_or_title_raises() -> None:
    with pytest.raises(MTEBConversionError, match="no text or abstract"):
        convert_retrieval_data_to_normalized_dataset(
            corpus={"doc-1": {"venue": "Conference"}},
            queries={"q-1": "query"},
            qrels={"q-1": {"doc-1": 1}},
        )


def test_query_string_value() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text"},
        queries={"q-1": "query text"},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.queries[0].text == "query text"


def test_query_dict_uses_text_field() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text"},
        queries={"q-1": {"text": "query text", "lang": "en"}},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.queries[0].text == "query text"
    assert dataset.queries[0].metadata == {"lang": "en"}


def test_query_dict_uses_query_field_when_text_missing() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text"},
        queries={"q-1": {"query": "query text"}},
        qrels={"q-1": {"doc-1": 1}},
    )

    assert dataset.queries[0].text == "query text"


def test_qrels_support_graded_relevance() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text"},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 2.0, "doc-2": 0.5}},
    )

    assert {(qrel.doc_id, qrel.relevance) for qrel in dataset.qrels} == {
        ("doc-1", 2.0),
        ("doc-2", 0.5),
    }


def test_qrels_keep_zero_relevance() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text", "doc-2": "other"},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1.0, "doc-2": 0.0}},
    )

    assert any(qrel.doc_id == "doc-2" and qrel.relevance == 0.0 for qrel in dataset.qrels)


def test_metadata_is_preserved_on_dataset() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus={"doc-1": "text"},
        queries={"q-1": "query"},
        qrels={"q-1": {"doc-1": 1}},
        metadata={"source": "mteb", "task_name": "SciFact"},
    )

    assert dataset.metadata == {"source": "mteb", "task_name": "SciFact"}


def test_ids_are_converted_to_strings() -> None:
    dataset = convert_retrieval_data_to_normalized_dataset(
        corpus=cast(dict[Any, Any], {1: "text"}),
        queries=cast(dict[Any, Any], {2: "query"}),
        qrels=cast(dict[Any, Any], {2: {1: 1}}),
    )

    assert dataset.corpus[0].doc_id == "1"
    assert dataset.queries[0].query_id == "2"
    assert dataset.qrels[0].query_id == "2"
    assert dataset.qrels[0].doc_id == "1"


@pytest.mark.parametrize(
    ("corpus", "queries"),
    [
        ({"doc-1": " "}, {"q-1": "query"}),
        ({"doc-1": "text"}, {"q-1": " "}),
    ],
)
def test_blank_text_triggers_validation_error(
    corpus: dict[str, str],
    queries: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        convert_retrieval_data_to_normalized_dataset(
            corpus=corpus,
            queries=queries,
            qrels={"q-1": {"doc-1": 1}},
        )
