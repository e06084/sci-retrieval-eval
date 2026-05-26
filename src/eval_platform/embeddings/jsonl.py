"""JSONL helpers for embedding records."""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Iterable
from typing import Any

from eval_platform.embeddings.schema import EmbeddingRecord

VECTOR_ENCODING = "float32_le_base64"
VECTOR_DTYPE = "float32"


class EmbeddingJSONLError(ValueError):
    """Raised when embeddings JSONL cannot be encoded or decoded."""


def _encode_vector(vector: list[float]) -> str:
    payload = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(payload).decode("ascii")


def _decode_vector(payload: str) -> list[float]:
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except Exception as exc:
        raise EmbeddingJSONLError("vector_b64 is not valid base64") from exc
    if len(raw) == 0 or len(raw) % 4 != 0:
        raise EmbeddingJSONLError("vector_b64 length must be a non-empty multiple of 4 bytes")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _dump_record(record: EmbeddingRecord) -> str:
    payload = {
        "chunk_id": record.chunk_id,
        "doc_id": record.doc_id,
        "vector_b64": _encode_vector(record.vector),
        "vector_encoding": VECTOR_ENCODING,
        "metadata": record.metadata,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _load_record(line: str) -> EmbeddingRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EmbeddingJSONLError("embedding JSONL line is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise EmbeddingJSONLError("embedding JSONL line must be an object")
    chunk_id = payload.get("chunk_id")
    doc_id = payload.get("doc_id")
    if not isinstance(chunk_id, str):
        raise EmbeddingJSONLError("chunk_id must be a string")
    if not isinstance(doc_id, str):
        raise EmbeddingJSONLError("doc_id must be a string")
    if payload.get("vector_encoding") != VECTOR_ENCODING:
        raise EmbeddingJSONLError(f"vector_encoding must be {VECTOR_ENCODING!r}")
    vector_b64 = payload.get("vector_b64")
    if not isinstance(vector_b64, str):
        raise EmbeddingJSONLError("vector_b64 must be a string")
    metadata: Any = payload.get("metadata") or {}
    return EmbeddingRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        vector=_decode_vector(vector_b64),
        metadata=metadata,
    )


def dump_embeddings_jsonl(records: Iterable[EmbeddingRecord]) -> str:
    """Serialize embedding records to JSONL text."""
    lines = [_dump_record(record) for record in records]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def load_embeddings_jsonl(text: str) -> list[EmbeddingRecord]:
    """Load embedding records from JSONL text."""
    records: list[EmbeddingRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        records.append(_load_record(line))
    return records
