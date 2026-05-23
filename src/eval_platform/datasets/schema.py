"""Normalized dataset record schemas."""

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class CorpusRecord(BaseModel):
    """A document in a normalized retrieval corpus."""

    doc_id: str
    title: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "text")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class QueryRecord(BaseModel):
    """A query in a normalized retrieval dataset."""

    query_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "text")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class QrelRecord(BaseModel):
    """A query-document relevance judgment."""

    query_id: str
    doc_id: str
    relevance: float = Field(default=1.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_id", "doc_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class NormalizedDataset(BaseModel):
    """In-memory container for a normalized retrieval dataset."""

    corpus: list[CorpusRecord]
    queries: list[QueryRecord]
    qrels: list[QrelRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)
