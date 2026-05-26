"""Retrieval client protocols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from eval_platform.retrieval.schema import RetrievalHit


class ElasticsearchRetrievalClient(Protocol):
    """Minimal Elasticsearch retrieval client used by the retrieval runner."""

    def search_bm25(
        self,
        index_name: str,
        query: str,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Return BM25 hits for one query."""
        ...

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Return hits enriched with chunk payloads, preserving input order."""
        ...


class MilvusRetrievalClient(Protocol):
    """Minimal Milvus retrieval client used by the retrieval runner."""

    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        """Return vector hits for one query vector."""
        ...


class RewriteClient(Protocol):
    """Optional query rewrite client."""

    def rewrite(self, query: str, max_queries: int) -> list[str]:
        """Return up to max_queries rewritten queries."""
        ...


class RerankClient(Protocol):
    """Optional rerank client."""

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Return reranked hits for one original query."""
        ...
