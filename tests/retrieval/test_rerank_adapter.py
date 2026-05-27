"""Tests for HTTP rerank adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from eval_platform.config import EndpointConfig, RerankConfig
from eval_platform.retrieval import (
    HTTPRerankClient,
    HTTPRerankClientConfig,
    RerankAdapterError,
    RetrievalHit,
    rerank_client_from_config,
    run_rerank_consistency_check,
)


class RecordingTransport:
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


class OrderedRerankClient:
    def __init__(self, order: list[int]) -> None:
        self._order = order

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        return [
            hits[index].model_copy(update={"score": float(len(hits) - rank)})
            for rank, index in enumerate(self._order[:top_n])
        ]


class FailingRerankClient:
    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        raise RerankAdapterError("boom")


def _hits() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            title="title 1",
            text="first text",
            rank=7,
            score=0.1,
            recall_source="hybrid",
            origin_es_score=1.1,
            origin_milvus_score=2.2,
            metadata={"source": "test"},
        ),
        RetrievalHit(
            chunk_id="chunk-2",
            doc_id="doc-2",
            title="title 2",
            text="second text",
            score=0.2,
            recall_source="hybrid",
            origin_es_score=1.2,
            origin_milvus_score=2.3,
            metadata={"source": "test-2"},
        ),
        RetrievalHit(
            chunk_id="chunk-3",
            doc_id="doc-3",
            title="title 3",
            text="third text",
            score=0.3,
            recall_source="hybrid",
        ),
    ]


def test_http_rerank_client_config_constructs() -> None:
    config = HTTPRerankClientConfig(
        endpoint_url="https://rerank.example/rerank",
        endpoint_id="endpoint-0",
        model_name="BAAI/bge-reranker-v2-m3",
        timeout_seconds=12.5,
        max_retries=1,
    )

    assert config.endpoint_url == "https://rerank.example/rerank"
    assert config.endpoint_id == "endpoint-0"
    assert config.model_name == "BAAI/bge-reranker-v2-m3"
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 1


def test_http_rerank_client_config_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        HTTPRerankClientConfig(endpoint_url="")
    with pytest.raises(ValidationError):
        HTTPRerankClientConfig(endpoint_url="https://example.com", endpoint_id=" ")
    with pytest.raises(ValidationError):
        HTTPRerankClientConfig(endpoint_url="https://example.com", model_name="")
    with pytest.raises(ValidationError):
        HTTPRerankClientConfig(endpoint_url="https://example.com", timeout_seconds=0)
    with pytest.raises(ValidationError):
        HTTPRerankClientConfig(endpoint_url="https://example.com", max_retries=-1)


def test_http_rerank_client_builds_request_and_uses_results_response() -> None:
    transport = RecordingTransport(
        [
            (
                200,
                b'{"results": ['
                b'{"index": 1, "relevance_score": 0.9},'
                b'{"index": 0, "relevance_score": 0.9},'
                b'{"index": 2, "relevance_score": 0.1}'
                b"]}",
            )
        ]
    )
    client = HTTPRerankClient(
        HTTPRerankClientConfig(
            endpoint_url="https://rerank.example/rerank",
            model_name="BAAI/bge-reranker-v2-m3",
            api_key="secret-key",
            timeout_seconds=9.0,
        ),
        transport=transport,
    )

    reranked = client.rerank("query text", _hits(), top_n=2)

    assert transport.calls == [
        {
            "endpoint_url": "https://rerank.example/rerank",
            "payload": {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "query text",
                "documents": ["first text", "second text", "third text"],
                "top_n": 3,
                "return_documents": False,
            },
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer secret-key",
            },
            "timeout_seconds": 9.0,
        }
    ]
    assert [hit.chunk_id for hit in reranked] == ["chunk-1", "chunk-2"]
    assert [hit.score for hit in reranked] == [0.9, 0.9]


def test_http_rerank_client_uses_data_response_and_local_top_n() -> None:
    transport = RecordingTransport(
        [
            (
                200,
                b'{"data": ['
                b'{"document_index": 2, "score": 0.3},'
                b'{"document_index": 0, "score": 0.8},'
                b'{"document_index": 1, "score": 0.5}'
                b"]}",
            )
        ]
    )
    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=transport,
    )

    reranked = client.rerank("query text", _hits(), top_n=2)

    assert "model" not in transport.calls[0]["payload"]
    assert [hit.chunk_id for hit in reranked] == ["chunk-1", "chunk-2"]
    assert [hit.score for hit in reranked] == [0.8, 0.5]


def test_http_rerank_client_skips_unusable_rows() -> None:
    body = json.dumps(
        {
            "results": [
                {"index": 10, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.9},
                {"index": 1},
                {"index": 2, "relevance_score": math.inf},
                {"index": 2, "relevance_score": 0.3},
            ]
        },
        allow_nan=True,
    ).encode("utf-8")
    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=RecordingTransport([(200, body)]),
    )

    reranked = client.rerank("query text", _hits(), top_n=3)

    assert [(hit.chunk_id, hit.score) for hit in reranked] == [
        ("chunk-3", 0.3),
        ("chunk-1", 0.1),
    ]


def test_http_rerank_client_preserves_hit_fields_and_does_not_set_rank() -> None:
    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=RecordingTransport(
            [(200, b'{"results": [{"index": 0, "relevance_score": 0.77}]}')]
        ),
    )

    reranked = client.rerank("query text", [_hits()[0]], top_n=1)

    assert reranked == [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            title="title 1",
            text="first text",
            rank=7,
            score=0.77,
            recall_source="hybrid",
            origin_es_score=1.1,
            origin_milvus_score=2.2,
            metadata={"source": "test"},
        )
    ]


def test_http_rerank_client_uses_blank_document_for_empty_text() -> None:
    transport = RecordingTransport(
        [(200, b'{"results": [{"index": 0, "relevance_score": 0.1}]}')]
    )
    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=transport,
    )

    client.rerank(
        "query text",
        [RetrievalHit(chunk_id="chunk-1", doc_id="doc-1", text="", score=1.0)],
        top_n=1,
    )

    assert transport.calls[0]["payload"]["documents"] == [" "]


@pytest.mark.parametrize("hits,top_n", [([], 10), (_hits(), 0), (_hits(), -1)])
def test_http_rerank_client_empty_input_or_non_positive_top_n_does_not_call_transport(
    hits: list[RetrievalHit],
    top_n: int,
) -> None:
    transport = RecordingTransport([])
    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=transport,
    )

    assert client.rerank("query text", hits, top_n=top_n) == []
    assert transport.calls == []


def test_http_rerank_client_errors_do_not_leak_api_key() -> None:
    client = HTTPRerankClient(
        HTTPRerankClientConfig(
            endpoint_url="https://rerank.example/rerank",
            api_key="secret-key",
        ),
        transport=RecordingTransport([(500, b'{"error": "boom"}')]),
    )

    with pytest.raises(RerankAdapterError) as exc_info:
        client.rerank("query text", _hits(), top_n=1)

    assert "HTTP 500" in str(exc_info.value)
    assert "secret-key" not in str(exc_info.value)


def test_http_rerank_client_invalid_json_and_empty_results_raise() -> None:
    invalid_client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=RecordingTransport([(200, b"{invalid-json}")]),
    )
    with pytest.raises(RerankAdapterError, match="invalid JSON"):
        invalid_client.rerank("query text", _hits(), top_n=1)

    empty_client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=RecordingTransport([(200, b'{"results": []}')]),
    )
    with pytest.raises(RerankAdapterError, match="usable results"):
        empty_client.rerank("query text", _hits(), top_n=1)


def test_http_rerank_client_transport_invalid_response_raises() -> None:
    def bad_transport(*args: Any, **kwargs: Any) -> tuple[int]:
        return (200,)

    client = HTTPRerankClient(
        HTTPRerankClientConfig(endpoint_url="https://rerank.example/rerank"),
        transport=bad_transport,
    )

    with pytest.raises(RerankAdapterError, match="invalid response"):
        client.rerank("query text", _hits(), top_n=1)


def test_rerank_client_from_config_selects_fixed_endpoint() -> None:
    client = rerank_client_from_config(
        RerankConfig(
            model="BAAI/bge-reranker-v2-m3",
            timeout_sec=7.0,
            max_retries=1,
            endpoints=[
                EndpointConfig(url="https://example.com/a", api_key="a-key"),
                EndpointConfig(url="https://example.com/b", api_key="b-key"),
            ],
        ),
        endpoint_index=1,
    )

    assert client._config.endpoint_url == "https://example.com/b"
    assert client._config.endpoint_id == "endpoint-1"
    assert client._config.model_name == "BAAI/bge-reranker-v2-m3"
    assert client._config.api_key == "b-key"
    assert client._config.timeout_seconds == 7.0
    assert client._config.max_retries == 1


def test_rerank_client_from_config_rejects_empty_or_out_of_range_endpoint() -> None:
    with pytest.raises(RerankAdapterError, match="must not be empty"):
        rerank_client_from_config(RerankConfig(endpoints=[]))
    with pytest.raises(RerankAdapterError, match="out of range"):
        rerank_client_from_config(
            RerankConfig(endpoints=[EndpointConfig(url="https://example.com/a")]),
            endpoint_index=1,
        )


def test_run_rerank_consistency_check_passes_for_same_ranking() -> None:
    result = run_rerank_consistency_check(
        [
            OrderedRerankClient([1, 0, 2]),
            OrderedRerankClient([1, 0, 2]),
        ],
        endpoint_ids=["endpoint-a", "endpoint-b"],
        query="query text",
        documents=["a", "b", "c"],
    )

    assert result.passed is True
    assert result.failure_reason is None
    assert result.rankings == {
        "endpoint-a": [1, 0, 2],
        "endpoint-b": [1, 0, 2],
    }


def test_run_rerank_consistency_check_fails_for_different_ranking() -> None:
    result = run_rerank_consistency_check(
        [
            OrderedRerankClient([1, 0, 2]),
            OrderedRerankClient([0, 1, 2]),
        ],
        endpoint_ids=["endpoint-a", "endpoint-b"],
        query="query text",
        documents=["a", "b", "c"],
    )

    assert result.passed is False
    assert "ranking differs" in (result.failure_reason or "")
    assert result.rankings == {
        "endpoint-a": [1, 0, 2],
        "endpoint-b": [0, 1, 2],
    }


def test_run_rerank_consistency_check_fails_when_client_errors() -> None:
    result = run_rerank_consistency_check(
        [
            OrderedRerankClient([1, 0, 2]),
            FailingRerankClient(),
        ],
        endpoint_ids=["endpoint-a", "endpoint-b"],
        query="query text",
        documents=["a", "b", "c"],
    )

    assert result.passed is False
    assert "Endpoint endpoint-b failed" in (result.failure_reason or "")
    assert result.rankings == {"endpoint-a": [1, 0, 2]}
