"""JSONL helpers for embedding records."""

from __future__ import annotations

from collections.abc import Iterable

from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl
from eval_platform.embeddings.schema import EmbeddingRecord


def dump_embeddings_jsonl(records: Iterable[EmbeddingRecord]) -> str:
    """Serialize embedding records to JSONL text."""
    return dump_jsonl(records)


def load_embeddings_jsonl(text: str) -> list[EmbeddingRecord]:
    """Load embedding records from JSONL text."""
    return load_jsonl(text, EmbeddingRecord)
