"""Tests for retrieval query path helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pytest

from eval_platform.retrieval.errors import RetrievalRunError
from eval_platform.retrieval.query_paths import (
    dedupe_queries,
    embed_query_paths,
    resolve_query_paths,
)


@dataclass
class QueryPathConfig:
    retrieval_mode: Literal["es", "milvus", "hybrid"] = "hybrid"
    rewrite_enabled: bool = False
    sub_queries: int = 0


class FakeRewriteClient:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites
        self.calls: list[tuple[str, int]] = []

    def rewrite(self, query: str, max_queries: int) -> list[str]:
        self.calls.append((query, max_queries))
        return self.rewrites


class FakeEmbeddingClient:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = [[float(index)] for index, _text in enumerate(texts)]
        return vectors[:-1] if self.mismatch else vectors


def test_dedupe_queries_filters_blanks_case_insensitively_and_caps() -> None:
    assert dedupe_queries(
        [" Alpha ", "", "alpha", "Beta", " beta ", "Gamma"],
        max_count=3,
    ) == ["Alpha", "Beta", "Gamma"]


def test_resolve_query_paths_rewrites_and_dedupes_to_sub_query_limit() -> None:
    rewrite = FakeRewriteClient([" ", "ALPHA QUERY", "beta query", "gamma query"])

    assert resolve_query_paths(
        " alpha query ",
        QueryPathConfig(rewrite_enabled=True, sub_queries=2),
        rewrite,
    ) == ["alpha query", "beta query", "gamma query"]
    assert rewrite.calls == [(" alpha query ", 2)]


def test_embed_query_paths_skips_embedding_for_es_mode() -> None:
    embedding = FakeEmbeddingClient()

    assert embed_query_paths(["alpha"], QueryPathConfig(retrieval_mode="es"), embedding) == []
    assert embedding.calls == []


def test_embed_query_paths_embeds_vector_modes_and_validates_count() -> None:
    embedding = FakeEmbeddingClient()

    assert embed_query_paths(
        ["alpha", "beta"],
        QueryPathConfig(retrieval_mode="milvus"),
        embedding,
    ) == [[0.0], [1.0]]
    assert embedding.calls == [["alpha", "beta"]]

    with pytest.raises(RetrievalRunError, match="different number of vectors"):
        embed_query_paths(
            ["alpha", "beta"],
            QueryPathConfig(retrieval_mode="hybrid"),
            FakeEmbeddingClient(mismatch=True),
        )
