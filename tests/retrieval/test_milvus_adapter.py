"""Tests for Milvus retrieval adapter."""

from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.config import MilvusConfig
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.retrieval import (
    HTTPElasticsearchRetrievalClient,
    HTTPElasticsearchRetrievalClientConfig,
    MilvusRetrievalAdapterError,
    PymilvusRetrievalClient,
    PymilvusRetrievalClientConfig,
    RetrievalRunConfig,
    milvus_retrieval_client_from_config,
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


class FakeMilvusClient:
    def __init__(self, result: list[list[dict[str, Any]]] | None = None) -> None:
        self.result = result or [
            [
                {
                    "id": "fallback-id",
                    "distance": 0.9,
                    "entity": {
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "title": "title",
                        "text": "text",
                        "chunk_index": 1,
                        "metadata": {"source": "milvus"},
                        "vector": [0.1, 0.2],
                    },
                }
            ]
        ]
        self.calls: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(kwargs)
        return self.result


def test_pymilvus_retrieval_client_calls_search_and_parses_hits() -> None:
    fake = FakeMilvusClient()
    client = PymilvusRetrievalClient(
        PymilvusRetrievalClientConfig(
            address="http://milvus.example:19530",
            search_params={"nprobe": 8},
        ),
        client=fake,
    )

    hits = client.search("chunks", [1.0, 2.0], 10)

    assert fake.calls == [
        {
            "collection_name": "chunks",
            "data": [[1.0, 2.0]],
            "anns_field": "vector",
            "limit": 10,
            "output_fields": [
                "chunk_id",
                "doc_id",
                "title",
                "text",
                "chunk_index",
                "start_offset",
                "end_offset",
                "metadata",
            ],
            "search_params": {"metric_type": "COSINE", "params": {"nprobe": 8}},
        }
    ]
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].doc_id == "doc-1"
    assert hits[0].score == 0.9
    assert hits[0].origin_milvus_score == 0.9
    assert hits[0].recall_source == "milvus"
    assert hits[0].metadata == {"source": "milvus", "chunk_index": 1}
    assert "vector" not in hits[0].metadata


def test_pymilvus_retrieval_client_falls_back_to_hit_id() -> None:
    client = PymilvusRetrievalClient(
        PymilvusRetrievalClientConfig(address="http://milvus.example:19530"),
        client=FakeMilvusClient(result=[[{"id": "chunk-id", "score": 0.5, "entity": {}}]]),
    )

    assert client.search("chunks", [1.0], 1)[0].chunk_id == "chunk-id"


def test_pymilvus_retrieval_client_requires_pymilvus(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymilvus":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(MilvusRetrievalAdapterError, match="pymilvus"):
        PymilvusRetrievalClient(PymilvusRetrievalClientConfig(address="http://milvus"))


def test_pymilvus_retrieval_client_config_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="address"):
        PymilvusRetrievalClientConfig(address="")
    with pytest.raises(ValueError, match="primary_key_field"):
        PymilvusRetrievalClientConfig(address="http://milvus", primary_key_field="")
    with pytest.raises(ValueError, match="vector_field"):
        PymilvusRetrievalClientConfig(address="http://milvus", vector_field="")


def test_milvus_retrieval_client_from_config_requires_address() -> None:
    with pytest.raises(MilvusRetrievalAdapterError, match="address"):
        milvus_retrieval_client_from_config(MilvusConfig())


class StaticEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_run_retrieval_hybrid_with_real_adapters_and_fakes(tmp_path: Path) -> None:
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
    es_client = HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(base_url="http://es.example"),
        transport=RecordingTransport(
                [
                    (
                        200,
                        {
                            "docs": [
                                {
                                    "_id": "chunk-1",
                                    "found": True,
                                    "_source": {"doc_id": "doc-1", "text": "milvus text"},
                                }
                            ]
                        },
                    ),
                    (
                        200,
                        {
                            "hits": {
                                "hits": [
                                    {
                                        "_id": "es-1",
                                        "_score": 1.0,
                                        "_source": {
                                            "chunk_id": "es-1",
                                            "doc_id": "doc-1",
                                            "text": "es text",
                                        },
                                    }
                                ]
                            }
                        },
                    ),
                (
                    200,
                    {
                        "docs": [
                            {
                                "_id": "chunk-1",
                                "found": True,
                                "_source": {"doc_id": "doc-1", "text": "milvus text"},
                            },
                            {
                                "_id": "es-1",
                                "found": True,
                                "_source": {"doc_id": "doc-1", "text": "es text"},
                            },
                        ]
                    },
                ),
            ]
        ),
    )
    milvus_client = PymilvusRetrievalClient(
        PymilvusRetrievalClientConfig(address="http://milvus.example:19530"),
        client=FakeMilvusClient(),
    )

    run_retrieval(
        store,
        store,
        RetrievalRunConfig(
            source_normalized_dataset_artifact_id="normalized-1",
            output_artifact_id="retrieval-1",
            retrieval_mode="hybrid",
            top_k=2,
            elasticsearch_index_artifact_id="es-artifact",
            milvus_collection_artifact_id="milvus-artifact",
            index_name="chunks",
            collection_name="chunks",
        ),
        es_client=es_client,
        milvus_client=milvus_client,
        embedding_client=StaticEmbeddingClient(),
    )

    records = read_retrieval_run_artifact(store, "retrieval-1")
    assert len(records[0].hits) == 2
    assert {hit.recall_source for hit in records[0].hits} == {"milvus", "es"}
