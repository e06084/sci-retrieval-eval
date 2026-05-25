"""Normalizer for the LitSearchRetrieval MTEB retrieval task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eval_platform.mteb_adapter.base import GenericRetrievalTaskNormalizer


def _has_usable_litsearch_text(payload: Any) -> bool:
    if isinstance(payload, str):
        return bool(payload.strip())
    if not isinstance(payload, Mapping):
        return False
    for field_name in ("text", "abstract", "title"):
        value = payload.get(field_name)
        if isinstance(value, str) and value.strip():
            return True
    return False


class LitSearchRetrievalNormalizer(GenericRetrievalTaskNormalizer):
    task_name = "LitSearchRetrieval"

    def extract_raw(
        self,
        task: Any,
        split: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
        # Reuse the shared extractor first, then apply LitSearch-specific cleanup.
        corpus, queries, qrels = super().extract_raw(task, split)

        filtered_corpus = {
            doc_id: payload
            for doc_id, payload in corpus.items()
            if _has_usable_litsearch_text(payload)
        }
        filtered_qrels = {
            query_id: {
                doc_id: relevance
                for doc_id, relevance in doc_relevances.items()
                if doc_id in filtered_corpus
            }
            for query_id, doc_relevances in qrels.items()
        }
        filtered_qrels = {
            query_id: doc_relevances
            for query_id, doc_relevances in filtered_qrels.items()
            if doc_relevances
        }
        filtered_queries = {
            query_id: payload
            for query_id, payload in queries.items()
            if query_id in filtered_qrels
        }
        return filtered_corpus, filtered_queries, filtered_qrels
