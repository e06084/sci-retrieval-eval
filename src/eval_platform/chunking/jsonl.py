"""JSONL helpers for chunk records."""

from collections.abc import Iterable

from eval_platform.chunking.schema import ChunkRecord
from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl


def dump_chunks_jsonl(chunks: Iterable[ChunkRecord]) -> str:
    """Serialize chunk records to JSONL text."""
    return dump_jsonl(chunks)


def load_chunks_jsonl(text: str) -> list[ChunkRecord]:
    """Load JSONL text into chunk records."""
    return load_jsonl(text, ChunkRecord)
