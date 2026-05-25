"""Convert MTEB-like retrieval data into normalized dataset records."""

from collections.abc import Mapping
from typing import Any

from eval_platform.datasets.schema import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
)


class MTEBConversionError(Exception):
    """Raised when MTEB retrieval data cannot be converted."""


def _corpus_text_from_mapping(payload: Mapping[str, Any]) -> str | None:
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    abstract = payload.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract
    return None


def _convert_corpus_record(doc_id: str, payload: Any) -> CorpusRecord:
    doc_id_str = str(doc_id)
    if isinstance(payload, str):
        return CorpusRecord(doc_id=doc_id_str, text=payload)

    if not isinstance(payload, Mapping):
        raise MTEBConversionError(
            f"Corpus document {doc_id_str} must be a string or mapping, got {type(payload)!r}"
        )

    text = _corpus_text_from_mapping(payload)
    if text is None:
        raise MTEBConversionError(
            f"Corpus document {doc_id_str} has no text or abstract field"
        )

    title = payload.get("title")
    title_value = title if isinstance(title, str) else None
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"title", "text", "abstract"}
    }
    return CorpusRecord(doc_id=doc_id_str, title=title_value, text=text, metadata=metadata)


def _convert_query_record(query_id: str, payload: Any) -> QueryRecord:
    query_id_str = str(query_id)
    if isinstance(payload, str):
        return QueryRecord(query_id=query_id_str, text=payload)

    if not isinstance(payload, Mapping):
        raise MTEBConversionError(
            f"Query {query_id_str} must be a string or mapping, got {type(payload)!r}"
        )

    text = payload.get("text")
    if not (isinstance(text, str) and text.strip()):
        query_text = payload.get("query")
        text = query_text if isinstance(query_text, str) else None

    if text is None:
        raise MTEBConversionError(f"Query {query_id_str} has no text or query field")

    metadata = {key: value for key, value in payload.items() if key not in {"text", "query"}}
    return QueryRecord(query_id=query_id_str, text=text, metadata=metadata)


def convert_retrieval_data_to_normalized_dataset(
    *,
    corpus: Mapping[str, Any],
    queries: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int | float]],
    metadata: dict[str, Any] | None = None,
) -> NormalizedDataset:
    """Convert generic retrieval task data into a normalized dataset."""
    corpus_records = [_convert_corpus_record(doc_id, payload) for doc_id, payload in corpus.items()]
    query_records = [
        _convert_query_record(query_id, payload) for query_id, payload in queries.items()
    ]

    qrel_records: list[QrelRecord] = []
    for query_id, doc_relevances in qrels.items():
        if not isinstance(doc_relevances, Mapping):
            raise MTEBConversionError(
                f"Qrels for query {query_id} must be a mapping, got {type(doc_relevances)!r}"
            )
        for doc_id, relevance in doc_relevances.items():
            qrel_records.append(
                QrelRecord(
                    query_id=str(query_id),
                    doc_id=str(doc_id),
                    relevance=float(relevance),
                )
            )

    return NormalizedDataset(
        corpus=corpus_records,
        queries=query_records,
        qrels=qrel_records,
        metadata=dict(metadata or {}),
    )
