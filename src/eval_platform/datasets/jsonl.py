"""JSONL helpers for dataset records."""

from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def dump_jsonl(records: Iterable[BaseModel]) -> str:
    """Serialize Pydantic records to JSONL text."""
    lines = [record.model_dump_json() for record in records]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def load_jsonl(text: str, model: type[T]) -> list[T]:
    """Load JSONL text into typed Pydantic records."""
    records: list[T] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        records.append(model.model_validate_json(line))
    return records
