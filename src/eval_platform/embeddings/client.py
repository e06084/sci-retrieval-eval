"""Embedding client protocol and fake implementation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for injectable embedding clients."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""


class FakeEmbeddingClient:
    """Deterministic local embedding client for tests."""

    def __init__(self, embedding_dim: int = 3) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        self._embedding_dim = embedding_dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values: list[float] = []
            for index in range(self._embedding_dim):
                start = (index * 4) % len(digest)
                chunk = digest[start : start + 4]
                if len(chunk) < 4:
                    chunk = chunk + digest[: 4 - len(chunk)]
                raw = int.from_bytes(chunk, byteorder="big", signed=False)
                values.append(raw / 4294967295.0)
            vectors.append(values)
        return vectors
