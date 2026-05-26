"""JSONL helpers for retrieval run records."""

from __future__ import annotations

from collections.abc import Iterable

from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl
from eval_platform.retrieval.schema import RetrievalQueryResult


def dump_retrieval_results_jsonl(records: Iterable[RetrievalQueryResult]) -> str:
    """Serialize retrieval query results to JSONL."""

    return dump_jsonl(records)


def load_retrieval_results_jsonl(text: str) -> list[RetrievalQueryResult]:
    """Load retrieval query results from JSONL."""

    return load_jsonl(text, RetrievalQueryResult)
