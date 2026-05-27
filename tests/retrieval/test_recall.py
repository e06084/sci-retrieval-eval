"""Tests for retrieval recall helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from eval_platform.retrieval.recall import recall_one
from eval_platform.retrieval.schema import RetrievalHit


@dataclass
class RecallConfig:
    retrieval_mode: Literal["es", "milvus", "hybrid"]
    top_k: int = 2
    index_name: str | None = "chunks-index"
    collection_name: str | None = "chunks-collection"
    hybrid_per_source_topk: int = 3
    rrf_path_topk: int = 2


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.enrich_calls: list[list[str]] = []

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        self.search_calls.append((index_name, query, top_k))
        return [
            RetrievalHit(
                chunk_id=f"es-{query}-{index}",
                doc_id=f"doc-es-{index}",
                text=f"es text {index}",
                score=10.0 - index,
                recall_source="es",
            )
            for index in range(1, top_k + 1)
        ]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        self.enrich_calls.append([hit.chunk_id for hit in hits])
        return [
            hit.model_copy(
                update={
                    "doc_id": hit.doc_id or f"doc-{hit.chunk_id}",
                    "text": hit.text or f"text {hit.chunk_id}",
                }
            )
            for hit in hits
        ]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float], int]] = []

    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        self.calls.append((collection_name, list(vector), top_k))
        return [
            RetrievalHit(
                chunk_id=f"mv-{index}",
                doc_id="",
                score=1.0 / index,
                recall_source="milvus",
            )
            for index in range(1, top_k + 1)
        ]


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


def test_recall_one_es_mode_does_not_call_embedding_or_milvus() -> None:
    es = FakeElasticsearchClient()
    milvus = FakeMilvusClient()
    embedding = FakeEmbeddingClient()

    hits, es_hits, milvus_hits, fused_hits = recall_one(
        query="alpha",
        config=RecallConfig(retrieval_mode="es"),
        es_client=es,
        milvus_client=milvus,
        embedding_client=embedding,
    )

    assert [hit.chunk_id for hit in hits] == ["es-alpha-1", "es-alpha-2"]
    assert hits == es_hits == fused_hits
    assert milvus_hits == []
    assert milvus.calls == []
    assert embedding.calls == []


def test_recall_one_milvus_mode_searches_and_enriches_without_bm25() -> None:
    es = FakeElasticsearchClient()
    milvus = FakeMilvusClient()

    hits, es_hits, milvus_hits, fused_hits = recall_one(
        query="alpha",
        config=RecallConfig(retrieval_mode="milvus"),
        es_client=es,
        milvus_client=milvus,
        embedding_client=None,
        vector=[0.1, 0.2],
    )

    assert es.search_calls == []
    assert milvus.calls == [("chunks-collection", [0.1, 0.2], 2)]
    assert es.enrich_calls == [["mv-1", "mv-2"]]
    assert es_hits == []
    assert [hit.chunk_id for hit in milvus_hits] == ["mv-1", "mv-2"]
    assert hits == fused_hits
    assert hits[0].text == "text mv-1"


def test_recall_one_hybrid_runs_milvus_bm25_rrf_and_enrich() -> None:
    es = FakeElasticsearchClient()
    milvus = FakeMilvusClient()

    hits, es_hits, milvus_hits, fused_hits = recall_one(
        query="alpha",
        config=RecallConfig(retrieval_mode="hybrid"),
        es_client=es,
        milvus_client=milvus,
        embedding_client=None,
        vector=[0.1, 0.2],
    )

    assert milvus.calls == [("chunks-collection", [0.1, 0.2], 3)]
    assert es.search_calls == [("chunks-index", "alpha", 3)]
    assert es.enrich_calls[0] == ["mv-1", "mv-2", "mv-3"]
    assert es.enrich_calls[1] == [hit.chunk_id for hit in fused_hits]
    assert [hit.chunk_id for hit in es_hits] == ["es-alpha-1", "es-alpha-2", "es-alpha-3"]
    assert [hit.chunk_id for hit in milvus_hits] == ["mv-1", "mv-2", "mv-3"]
    assert hits == fused_hits
    assert len(hits) == 2
