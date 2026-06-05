"""Tests for retrieval run orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking.progress import ProgressEvent
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.retrieval import (
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    HTTPRerankClient,
    HTTPRerankClientConfig,
    RetrievalHit,
    RetrievalQueryResult,
    RetrievalRunConfig,
    RetrievalRunError,
    read_retrieval_run_artifact,
    run_retrieval,
    write_retrieval_run_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    artifact_store = LocalArtifactStore(tmp_path)
    write_normalized_dataset_artifact(
        artifact_store,
        "normalized-1",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="doc")],
            queries=[
                QueryRecord(query_id="q-1", text="alpha query"),
                QueryRecord(query_id="q-2", text="error query"),
            ],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1")],
        ),
    )
    return artifact_store


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index), float(len(text))] for index, text in enumerate(texts, start=1)]


class FakeElasticsearchClient:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.enrich_calls: list[list[str]] = []

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        self.search_calls.append((index_name, query, top_k))
        if query == "error query":
            raise RuntimeError("boom")
        return [
            RetrievalHit(
                chunk_id=f"es-{query}-{index}",
                doc_id=f"doc-es-{index}",
                title=f"title {index}",
                text=f"es text {query} {index}",
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
                    "title": hit.title or f"title {hit.chunk_id}",
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
                doc_id=f"doc-mv-{index}",
                score=1.0 / index,
                recall_source="milvus",
            )
            for index in range(1, top_k + 1)
        ]


class FakeRewriteClient:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites
        self.calls: list[tuple[str, int]] = []

    def rewrite(self, query: str, max_queries: int) -> list[str]:
        self.calls.append((query, max_queries))
        return self.rewrites


class FakeRerankClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        self.calls.append((query, [hit.chunk_id for hit in hits], top_n))
        return [
            hit.model_copy(update={"score": 100.0 - index})
            for index, hit in enumerate(reversed(hits[:top_n]), start=1)
        ]


class RecordingRerankTransport:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        endpoint_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> tuple[int, bytes]:
        self.calls.append(
            {
                "endpoint_url": endpoint_url,
                "payload": json.loads(payload.decode("utf-8")),
                "headers": dict(headers),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._responses[len(self.calls) - 1]


def test_retrieval_run_config_uses_sciverse_v1_defaults() -> None:
    config = RetrievalRunConfig(
        source_normalized_dataset_artifact_id="normalized-1",
        output_artifact_id="retrieval-1",
        retrieval_mode="hybrid",
        elasticsearch_index_artifact_id="es-artifact",
        milvus_collection_artifact_id="milvus-artifact",
    )

    assert config.top_k == 100
    assert config.hybrid_per_source_topk == 50
    assert config.rrf_path_topk == 25
    assert config.rerank_cross_path_topk == 50
    assert config.rerank_candidate_cap == 0


def _config(**overrides: Any) -> RetrievalRunConfig:
    payload: dict[str, Any] = {
        "source_normalized_dataset_artifact_id": "normalized-1",
        "output_artifact_id": "retrieval-1",
        "retrieval_mode": "hybrid",
        "top_k": 2,
        "query_limit": 1,
        "elasticsearch_index_artifact_id": "es-artifact",
        "milvus_collection_artifact_id": "milvus-artifact",
        "index_name": "chunks-index",
        "collection_name": "chunks-collection",
        "hybrid_per_source_topk": 3,
        "rrf_path_topk": 2,
    }
    payload.update(overrides)
    if payload.get("execution_mode") == "replay" and "trace_mode" not in overrides:
        payload["trace_mode"] = "replay"
    return RetrievalRunConfig(**payload)


def test_run_retrieval_es_mode_uses_only_elasticsearch(store: LocalArtifactStore) -> None:
    es = FakeElasticsearchClient()
    embedding = FakeEmbeddingClient()
    milvus = FakeMilvusClient()

    manifest = run_retrieval(
        store,
        store,
        _config(retrieval_mode="es", milvus_collection_artifact_id=None),
        es_client=es,
        embedding_client=embedding,
        milvus_client=milvus,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert len(es.search_calls) == 1
    assert embedding.calls == []
    assert milvus.calls == []
    assert [hit.rank for hit in records[0].hits] == [1, 2]
    assert manifest.metadata["retrieval_mode"] == "es"


def test_run_retrieval_milvus_mode_embeds_searches_and_enriches(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    embedding = FakeEmbeddingClient()
    milvus = FakeMilvusClient()

    run_retrieval(
        store,
        store,
        _config(retrieval_mode="milvus"),
        es_client=es,
        embedding_client=embedding,
        milvus_client=milvus,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert embedding.calls == [["alpha query"]]
    assert milvus.calls[0][0] == "chunks-collection"
    assert es.search_calls == []
    assert es.enrich_calls == [["mv-1", "mv-2"]]
    assert records[0].hits[0].text == ""  # text stripped from artifact


def test_run_retrieval_hybrid_runs_rrf_and_enriches(store: LocalArtifactStore) -> None:
    es = FakeElasticsearchClient()
    embedding = FakeEmbeddingClient()
    milvus = FakeMilvusClient()

    manifest = run_retrieval(
        store,
        store,
        _config(retrieval_mode="hybrid"),
        es_client=es,
        embedding_client=embedding,
        milvus_client=milvus,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert embedding.calls == [["alpha query"]]
    assert milvus.calls[0][2] == 3
    assert es.search_calls[0] == ("chunks-index", "alpha query", 3)
    assert records[0].hits[0].rank == 1
    assert [(dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("elasticsearch_index", "es-artifact"),
        ("milvus_collection", "milvus-artifact"),
    ]
    assert manifest.metadata["query_count"] == 1
    assert manifest.metadata["result_record_count"] == 1


def test_run_retrieval_reports_query_progress(store: LocalArtifactStore) -> None:
    events: list[ProgressEvent] = []

    run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            query_limit=2,
        ),
        es_client=FakeElasticsearchClient(),
        progress_reporter=events.append,
    )

    retrieval_events = [event for event in events if event.stage == "retrieval_run"]

    assert [(event.current, event.total) for event in retrieval_events] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert retrieval_events[1].metadata["query_id"] == "q-1"
    assert retrieval_events[2].metadata["query_id"] == "q-2"
    assert retrieval_events[2].metadata["query_error"] == "boom"
    assert retrieval_events[2].metadata["failed_query_count"] == 1


def test_run_retrieval_defaults_to_replay_trace(store: LocalArtifactStore) -> None:
    es = FakeElasticsearchClient()

    manifest = run_retrieval(
        store,
        store,
        _config(retrieval_mode="es", milvus_collection_artifact_id=None),
        es_client=es,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert manifest.metadata["trace_mode"] == "light"
    assert manifest.metadata["execution_mode"] == "live"
    assert "include_trace" not in manifest.metadata
    assert records[0].trace is not None
    assert records[0].trace["rewrite_queries"] == ["alpha query"]
    assert records[0].trace["per_query"][0]["es_hits"]
    assert len(records[0].trace["final_hits"]) == 2
    assert all("doc_id" in hit for hit in records[0].trace["final_hits"])


def test_run_retrieval_trace_mode_none_omits_trace(store: LocalArtifactStore) -> None:
    es = FakeElasticsearchClient()

    manifest = run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            trace_mode="none",
        ),
        es_client=es,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert manifest.metadata["trace_mode"] == "none"
    assert "include_trace" not in manifest.metadata
    assert records[0].trace is None


def test_run_retrieval_rewrite_dedupes_and_batches_embedding(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    embedding = FakeEmbeddingClient()
    milvus = FakeMilvusClient()
    rewrite = FakeRewriteClient([" ", "ALPHA QUERY", "beta query", "beta query", "gamma query"])

    run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="milvus",
            rewrite_enabled=True,
            sub_queries=2,
        ),
        es_client=es,
        embedding_client=embedding,
        milvus_client=milvus,
        rewrite_client=rewrite,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert embedding.calls == [["alpha query", "beta query", "gamma query"]]
    assert records[0].trace is not None
    assert records[0].trace["rewrite_queries"] == ["alpha query", "beta query", "gamma query"]


def test_run_retrieval_rerank_caps_head_and_preserves_tail(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    rerank = FakeRerankClient()

    run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            top_k=3,
            rerank_enabled=True,
            rerank_candidate_cap=2,
            rerank_cross_path_topk=2,
        ),
        es_client=es,
        rerank_client=rerank,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert rerank.calls == [("alpha query", ["es-alpha query-1", "es-alpha query-2"], 2)]
    assert [hit.chunk_id for hit in records[0].hits] == [
        "es-alpha query-2",
        "es-alpha query-1",
        "es-alpha query-3",
    ]
    assert records[0].trace is not None
    assert [hit["chunk_id"] for hit in records[0].trace["rerank_input"]] == [
        "es-alpha query-1",
        "es-alpha query-2",
    ]


def test_run_retrieval_paper_cap_zero_preserves_default_behavior(
    store: LocalArtifactStore,
) -> None:
    base_es = FakeElasticsearchClient()
    base_embedding = FakeEmbeddingClient()
    base_milvus = FakeMilvusClient()
    run_retrieval(
        store,
        store,
        _config(output_artifact_id="no-cap", retrieval_mode="hybrid"),
        es_client=base_es,
        embedding_client=base_embedding,
        milvus_client=base_milvus,
    )
    base_records = read_retrieval_run_artifact(store, "no-cap")

    cap_es = FakeElasticsearchClient()
    cap_embedding = FakeEmbeddingClient()
    cap_milvus = FakeMilvusClient()
    run_retrieval(
        store,
        store,
        _config(output_artifact_id="zero-cap", retrieval_mode="hybrid", paper_cap=0),
        es_client=cap_es,
        embedding_client=cap_embedding,
        milvus_client=cap_milvus,
    )
    cap_records = read_retrieval_run_artifact(store, "zero-cap")

    assert [record.model_dump(mode="json") for record in cap_records] == [
        record.model_dump(mode="json") for record in base_records
    ]


def test_run_retrieval_paper_cap_limits_chunks_per_doc_before_rerank(
    store: LocalArtifactStore,
) -> None:
    class RepeatedDocESClient(FakeElasticsearchClient):
        def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
            rows = [
                RetrievalHit(
                    chunk_id="es-a1",
                    doc_id="doc-a",
                    text="a1",
                    score=10.0,
                    recall_source="es",
                ),
                RetrievalHit(
                    chunk_id="es-b1",
                    doc_id="doc-b",
                    text="b1",
                    score=9.0,
                    recall_source="es",
                ),
                RetrievalHit(
                    chunk_id="es-c1",
                    doc_id="doc-c",
                    text="c1",
                    score=8.0,
                    recall_source="es",
                ),
            ]
            return rows[:top_k]

        def enrich_by_chunk_ids(
            self,
            index_name: str,
            hits: Sequence[RetrievalHit],
        ) -> list[RetrievalHit]:
            return list(hits)

    class RepeatedDocMilvusClient(FakeMilvusClient):
        def search(
            self,
            collection_name: str,
            vector: Sequence[float],
            top_k: int,
        ) -> list[RetrievalHit]:
            rows = [
                RetrievalHit(
                    chunk_id="mv-a1",
                    doc_id="doc-a",
                    score=1.0,
                    recall_source="milvus",
                ),
                RetrievalHit(
                    chunk_id="mv-a2",
                    doc_id="doc-a",
                    score=0.9,
                    recall_source="milvus",
                ),
                RetrievalHit(
                    chunk_id="mv-b1",
                    doc_id="doc-b",
                    score=0.8,
                    recall_source="milvus",
                ),
            ]
            return rows[:top_k]

    rerank = FakeRerankClient()
    run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="paper-cap-rerank",
            retrieval_mode="hybrid",
            top_k=3,
            hybrid_per_source_topk=3,
            rrf_path_topk=3,
            paper_cap=1,
            rerank_enabled=True,
            rerank_candidate_cap=3,
            rerank_cross_path_topk=3,
        ),
        es_client=RepeatedDocESClient(),
        embedding_client=FakeEmbeddingClient(),
        milvus_client=RepeatedDocMilvusClient(),
        rerank_client=rerank,
    )
    records = read_retrieval_run_artifact(store, "paper-cap-rerank")
    trace = records[0].trace

    assert trace is not None
    capped_doc_ids = [row["doc_id"] for row in trace["paper_capped_hits"]]
    rerank_input_doc_ids = [row["doc_id"] for row in trace["rerank_input"]]
    assert capped_doc_ids == rerank_input_doc_ids
    assert len(capped_doc_ids) == len(set(capped_doc_ids))
    assert "doc-a" in capped_doc_ids
    assert "doc-b" in capped_doc_ids


def test_run_retrieval_rerank_enabled_with_http_rerank_client(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    transport = RecordingRerankTransport(
        [
            (
                200,
                b'{"results": ['
                b'{"index": 1, "relevance_score": 0.95},'
                b'{"index": 0, "relevance_score": 0.10}'
                b"]}",
            )
        ]
    )
    rerank = HTTPRerankClient(
        HTTPRerankClientConfig(
            endpoint_url="https://rerank.example/rerank",
            model_name="BAAI/bge-reranker-v2-m3",
        ),
        transport=transport,
    )

    run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            top_k=2,
            rerank_enabled=True,
            rerank_cross_path_topk=2,
        ),
        es_client=es,
        rerank_client=rerank,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert transport.calls[0]["endpoint_url"] == "https://rerank.example/rerank"
    assert transport.calls[0]["payload"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert transport.calls[0]["payload"]["documents"] == [
        "es text alpha query 1",
        "es text alpha query 2",
    ]
    assert [hit.chunk_id for hit in records[0].hits] == [
        "es-alpha query-2",
        "es-alpha query-1",
    ]
    assert [hit.rank for hit in records[0].hits] == [1, 2]
    assert [hit.score for hit in records[0].hits] == [0.95, 0.1]


def test_run_retrieval_records_query_error_and_continues(store: LocalArtifactStore) -> None:
    es = FakeElasticsearchClient()

    manifest = run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            query_limit=None,
        ),
        es_client=es,
    )
    records = read_retrieval_run_artifact(store, "retrieval-1")

    assert len(records) == 2
    assert records[0].error is None
    assert records[1].error == "boom"
    assert records[1].trace is not None
    assert records[1].trace["error"] == "boom"
    assert records[1].trace["error_stage"] == "unknown"
    assert records[1].trace["rewrite_queries"] == ["error query"]
    assert records[1].trace["final_hits"] == []
    assert manifest.metadata["failed_query_count"] == 1
    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "retrieval-1") is True


def test_run_retrieval_run_with_failed_query_can_be_replayed(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    run_retrieval(
        store,
        store,
        _config(
            retrieval_mode="es",
            milvus_collection_artifact_id=None,
            query_limit=None,
        ),
        es_client=es,
    )

    replay_manifest = run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="replayed-run",
            execution_mode="replay",
            replay_source_retrieval_run_artifact_id="retrieval-1",
        ),
    )
    source_records = read_retrieval_run_artifact(store, "retrieval-1")
    replay_records = read_retrieval_run_artifact(store, "replayed-run")

    assert [record.model_dump(mode="json") for record in replay_records] == [
        record.model_dump(mode="json") for record in source_records
    ]
    assert replay_manifest.metadata["failed_query_count"] == 1


def test_run_retrieval_fixed_live_artifact_can_be_replayed_without_behavior_changes(
    store: LocalArtifactStore,
) -> None:
    es = FakeElasticsearchClient()
    embedding = FakeEmbeddingClient()
    milvus = FakeMilvusClient()
    rewrite = FakeRewriteClient(["ALPHA QUERY", "beta query", "gamma query"])
    rerank = FakeRerankClient()

    live_manifest = run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="contract-live",
            retrieval_mode="hybrid",
            top_k=2,
            query_limit=1,
            rewrite_enabled=True,
            sub_queries=2,
            rerank_enabled=True,
            rerank_candidate_cap=2,
            rerank_cross_path_topk=2,
            trace_mode="replay",
            execution_mode="live",
        ),
        es_client=es,
        embedding_client=embedding,
        milvus_client=milvus,
        rewrite_client=rewrite,
        rerank_client=rerank,
    )
    live_records = read_retrieval_run_artifact(store, "contract-live")

    replay_manifest = run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="contract-replay",
            retrieval_mode="hybrid",
            top_k=2,
            query_limit=1,
            rewrite_enabled=True,
            sub_queries=2,
            rerank_enabled=True,
            rerank_candidate_cap=2,
            rerank_cross_path_topk=2,
            trace_mode="replay",
            execution_mode="replay",
            replay_source_retrieval_run_artifact_id="contract-live",
        ),
    )
    replay_records = read_retrieval_run_artifact(store, "contract-replay")

    assert [record.model_dump(mode="json") for record in replay_records] == [
        record.model_dump(mode="json") for record in live_records
    ]
    assert live_records[0].trace is not None
    assert live_records[0].trace["rewrite_queries"] == [
        "alpha query",
        "beta query",
        "gamma query",
    ]
    assert [hit.chunk_id for hit in live_records[0].hits] == [
        "es-beta query-1",
        "es-alpha query-1",
    ]
    assert live_manifest.metadata["retrieval_mode"] == "hybrid"
    assert live_manifest.metadata["trace_mode"] == "replay"
    assert live_manifest.metadata["execution_mode"] == "live"
    assert live_manifest.metadata["rerank_candidate_cap"] == 2
    assert live_manifest.metadata["query_count"] == 1
    assert replay_manifest.metadata["execution_mode"] == "replay"
    assert replay_manifest.metadata["replay_source_retrieval_run_artifact_id"] == (
        "contract-live"
    )
    assert [(dep.artifact_type, dep.artifact_id) for dep in live_manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("elasticsearch_index", "es-artifact"),
        ("milvus_collection", "milvus-artifact"),
    ]
    assert [(dep.artifact_type, dep.artifact_id) for dep in replay_manifest.dependencies] == [
        ("normalized_dataset", "normalized-1"),
        ("retrieval_run", "contract-live"),
    ]
    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "contract-live") is True
    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "contract-replay") is True


def test_run_retrieval_requires_missing_clients_before_writing(
    store: LocalArtifactStore,
) -> None:
    with pytest.raises(RetrievalRunError, match="embedding_client"):
        run_retrieval(
            store,
            store,
            _config(retrieval_mode="milvus"),
            es_client=FakeElasticsearchClient(),
            milvus_client=FakeMilvusClient(),
        )

    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "retrieval-1") is False


def test_run_retrieval_replay_copies_source_results(store: LocalArtifactStore) -> None:
    source_records = [
        RetrievalQueryResult(
            query_id="q-1",
            query_text="alpha query",
            hits=[
                RetrievalHit(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="text",
                    score=1.0,
                    recall_source="hybrid",
                )
            ],
            trace={
                "rewrite_queries": ["alpha query"],
                "per_query": [{"query": "alpha query", "es_hits": [], "milvus_hits": []}],
                "rerank_input": [],
                "rerank_hits": [],
                "final_hits": [{"rank": 1, "chunk_id": "chunk-1"}],
            },
        )
    ]
    write_retrieval_run_artifact(store, "source-run", source_records)

    manifest = run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="replayed-run",
            execution_mode="replay",
            replay_source_retrieval_run_artifact_id="source-run",
        ),
    )
    replayed = read_retrieval_run_artifact(store, "replayed-run")

    assert [record.model_dump(mode="json") for record in replayed] == [
        record.model_dump(mode="json") for record in source_records
    ]
    assert manifest.metadata["execution_mode"] == "replay"
    assert manifest.metadata["replay_source_retrieval_run_artifact_id"] == "source-run"
    assert ("retrieval_run", "source-run") in [
        (dep.artifact_type, dep.artifact_id) for dep in manifest.dependencies
    ]


def test_run_retrieval_replay_does_not_call_clients(store: LocalArtifactStore) -> None:
    source_records = [
        RetrievalQueryResult(
            query_id="q-1",
            query_text="alpha query",
            hits=[],
            trace={"rewrite_queries": ["alpha query"], "per_query": [], "final_hits": []},
        )
    ]
    write_retrieval_run_artifact(store, "source-run", source_records)
    es = FakeElasticsearchClient()
    milvus = FakeMilvusClient()
    embedding = FakeEmbeddingClient()
    rewrite = FakeRewriteClient(["beta query"])
    rerank = FakeRerankClient()

    run_retrieval(
        store,
        store,
        _config(
            output_artifact_id="replayed-run",
            execution_mode="replay",
            replay_source_retrieval_run_artifact_id="source-run",
            rewrite_enabled=True,
            rerank_enabled=True,
        ),
        es_client=es,
        milvus_client=milvus,
        embedding_client=embedding,
        rewrite_client=rewrite,
        rerank_client=rerank,
    )

    assert es.search_calls == []
    assert es.enrich_calls == []
    assert milvus.calls == []
    assert embedding.calls == []
    assert rewrite.calls == []
    assert rerank.calls == []


def test_run_retrieval_replay_fails_when_source_lacks_trace(
    store: LocalArtifactStore,
) -> None:
    write_retrieval_run_artifact(
        store,
        "source-run",
        [RetrievalQueryResult(query_id="q-1", query_text="alpha query", hits=[])],
    )

    with pytest.raises(RetrievalRunError, match="missing replay trace"):
        run_retrieval(
            store,
            store,
            _config(
                output_artifact_id="replayed-run",
                execution_mode="replay",
                replay_source_retrieval_run_artifact_id="source-run",
            ),
        )

    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "replayed-run") is False


def test_retrieval_run_config_replay_requires_source_artifact_id() -> None:
    with pytest.raises(ValueError, match="replay_source_retrieval_run_artifact_id"):
        _config(execution_mode="replay")
