"""Tests for Elasticsearch retrieval adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any
from urllib.request import Request

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.config import ElasticsearchConfig
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.retrieval import (
    ElasticsearchRetrievalAdapterError,
    HTTPElasticsearchRetrievalClient,
    HTTPElasticsearchRetrievalClientConfig,
    RetrievalHit,
    RetrievalRunConfig,
    elasticsearch_retrieval_client_from_config,
    read_retrieval_run_artifact,
    run_retrieval,
)


class RecordingTransport:
    def __init__(self, responses: list[tuple[int, dict[str, Any] | bytes]]) -> None:
        self.responses = responses
        self.calls: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> tuple[int, bytes]:
        self.calls.append(request)
        status, payload = self.responses.pop(0)
        if isinstance(payload, bytes):
            return status, payload
        return status, json.dumps(payload).encode("utf-8")


def _request_body(request: Request) -> dict[str, Any]:
    data = request.data
    assert isinstance(data, bytes)
    return json.loads(data.decode("utf-8"))


def test_search_bm25_builds_expected_request_and_parses_hits() -> None:
    transport = RecordingTransport(
        [
            (
                200,
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "fallback-id",
                                "_score": 3.5,
                                "_source": {
                                    "chunk_id": "chunk-1",
                                    "doc_id": "doc-1",
                                    "title": "title",
                                    "text": "text",
                                    "chunk_index": 7,
                                    "metadata": {"source": "test"},
                                },
                            }
                        ]
                    }
                },
            )
        ]
    )
    client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(
            base_url="http://es.example/",
            username="user",
            password="secret",
        ),
        transport=transport,
    )

    hits = client.search_bm25("chunks", "alpha", 5)

    assert transport.calls[0].full_url == "http://es.example/chunks/_search"
    assert transport.calls[0].get_method() == "POST"
    expected_token = base64.b64encode(b"user:secret").decode("ascii")
    assert transport.calls[0].get_header("Authorization") == f"Basic {expected_token}"
    body = _request_body(transport.calls[0])
    assert body["size"] == 5
    assert body["query"]["multi_match"]["query"] == "alpha"
    assert body["query"]["multi_match"]["fields"] == ["title^1.5", "text"]
    assert body["sort"] == [{"_score": {"order": "desc"}}, {"chunk_id": {"order": "asc"}}]
    assert hits == [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            title="title",
            text="text",
            score=3.5,
            recall_source="es",
            origin_es_score=3.5,
            metadata={"source": "test", "chunk_index": 7},
        )
    ]


def test_search_bm25_falls_back_to_id_for_missing_chunk_id() -> None:
    client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(base_url="http://es.example"),
        transport=RecordingTransport(
            [(200, {"hits": {"hits": [{"_id": "chunk-fallback", "_score": 1.0, "_source": {}}]}})]
        ),
    )

    assert client.search_bm25("chunks", "alpha", 1)[0].chunk_id == "chunk-fallback"


def test_enrich_by_chunk_ids_preserves_order_and_scores() -> None:
    transport = RecordingTransport(
        [
            (
                200,
                {
                    "docs": [
                        {
                            "_id": "chunk-2",
                            "found": True,
                            "_source": {"doc_id": "doc-2", "text": "two"},
                        },
                        {
                            "_id": "chunk-1",
                            "found": True,
                            "_source": {
                                "doc_id": "doc-1",
                                "title": "one",
                                "text": "one text",
                                "start_offset": 10,
                            },
                        },
                        {"_id": "chunk-missing", "found": False},
                    ]
                },
            )
        ]
    )
    client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(base_url="http://es.example"),
        transport=transport,
    )
    hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="",
            score=10.0,
            recall_source="milvus",
            origin_milvus_score=0.8,
        ),
        RetrievalHit(chunk_id="chunk-missing", doc_id="", score=9.0, recall_source="milvus"),
    ]

    enriched = client.enrich_by_chunk_ids("chunks", hits)

    assert [hit.chunk_id for hit in enriched] == ["chunk-1", "chunk-missing"]
    assert enriched[0].score == 10.0
    assert enriched[0].recall_source == "milvus"
    assert enriched[0].origin_milvus_score == 0.8
    assert enriched[0].doc_id == "doc-1"
    assert enriched[0].metadata["start_offset"] == 10
    assert enriched[1].metadata["enrich_missing"] is True
    assert _request_body(transport.calls[0])["ids"] == ["chunk-1", "chunk-missing"]


def test_http_error_and_invalid_json_raise_without_leaking_password() -> None:
    client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(
            base_url="http://es.example",
            username="user",
            password="super-secret",
        ),
        transport=RecordingTransport([(500, {"error": "failed"})]),
    )

    with pytest.raises(ElasticsearchRetrievalAdapterError) as exc_info:
        client.search_bm25("chunks", "alpha", 1)
    assert "super-secret" not in str(exc_info.value)

    invalid_client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(base_url="http://es.example"),
        transport=RecordingTransport([(200, b"not json")]),
    )
    with pytest.raises(ElasticsearchRetrievalAdapterError, match="not valid JSON"):
        invalid_client.search_bm25("chunks", "alpha", 1)


def test_elasticsearch_retrieval_client_from_config_requires_url() -> None:
    with pytest.raises(ElasticsearchRetrievalAdapterError, match="url"):
        elasticsearch_retrieval_client_from_config(ElasticsearchConfig())


def test_run_retrieval_es_mode_with_http_adapter_fake_transport(tmp_path: Any) -> None:
    store = LocalArtifactStore(tmp_path)
    write_normalized_dataset_artifact(
        store,
        "normalized-1",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="doc text")],
            queries=[QueryRecord(query_id="q-1", text="alpha")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1")],
        ),
    )
    client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(base_url="http://es.example"),
        transport=RecordingTransport(
            [
                (
                    200,
                    {
                        "hits": {
                            "hits": [
                                {
                                    "_id": "chunk-1",
                                    "_score": 2.0,
                                    "_source": {
                                        "chunk_id": "chunk-1",
                                        "doc_id": "doc-1",
                                        "text": "chunk text",
                                    },
                                }
                            ]
                        }
                    },
                )
            ]
        ),
    )

    run_retrieval(
        store,
        store,
        RetrievalRunConfig(
            source_normalized_dataset_artifact_id="normalized-1",
            output_artifact_id="retrieval-1",
            retrieval_mode="es",
            top_k=1,
            elasticsearch_index_artifact_id="es-artifact",
            index_name="chunks",
        ),
        es_client=client,
    )

    records = read_retrieval_run_artifact(store, "retrieval-1")
    assert records[0].hits[0].doc_id == "doc-1"


class StaticEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]
