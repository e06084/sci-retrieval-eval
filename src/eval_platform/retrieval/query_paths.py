"""Query rewrite, dedupe, and embedding helpers for retrieval runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from eval_platform.embeddings import EmbeddingClient
from eval_platform.retrieval.clients import RewriteClient
from eval_platform.retrieval.errors import RetrievalRunError


class QueryPathConfig(Protocol):
    retrieval_mode: Literal["es", "milvus", "hybrid"]
    rewrite_enabled: bool
    sub_queries: int


def resolve_query_paths(
    query_text: str,
    config: QueryPathConfig,
    rewrite_client: RewriteClient | None,
) -> list[str]:
    """Return the original query plus deduped rewrite paths."""

    queries = [query_text.strip()]
    if config.rewrite_enabled and config.sub_queries > 0:
        if rewrite_client is None:
            raise RetrievalRunError("rewrite_client is required when rewrite is enabled")
        queries.extend(rewrite_client.rewrite(query_text, config.sub_queries))
    return dedupe_queries(queries, max_count=1 + config.sub_queries)


def dedupe_queries(values: Sequence[str], *, max_count: int) -> list[str]:
    """Trim, drop blanks, and case-insensitively dedupe query paths."""

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = value.strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= max_count:
            break
    return out


def embed_query_paths(
    queries: list[str],
    config: QueryPathConfig,
    embedding_client: EmbeddingClient | None,
) -> list[list[float]]:
    """Embed query paths only for vector-backed retrieval modes."""

    if config.retrieval_mode not in {"milvus", "hybrid"}:
        return []
    if embedding_client is None:
        raise RetrievalRunError("embedding_client is required for milvus/hybrid retrieval")
    vectors = embedding_client.embed_texts(queries)
    if len(vectors) != len(queries):
        raise RetrievalRunError("embedding client returned a different number of vectors")
    return vectors
