"""Tests for HTTP embedding client."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import ValidationError

from eval_platform.embeddings import (
    EmbeddingClientError,
    HTTPEmbeddingClient,
    HTTPEmbeddingClientConfig,
    http_embedding_client_from_env,
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


def test_http_embedding_client_config_constructs() -> None:
    config = HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed")
    assert config.endpoint_url == "https://example.com/embed"
    assert config.timeout_seconds == 60.0
    assert config.batch_size == 32


def test_http_embedding_client_config_rejects_empty_endpoint() -> None:
    with pytest.raises(ValidationError):
        HTTPEmbeddingClientConfig(endpoint_url="")


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_http_embedding_client_config_rejects_non_positive_timeout(value: float) -> None:
    with pytest.raises(ValidationError):
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com", timeout_seconds=value)


@pytest.mark.parametrize("value", [0, -1])
def test_http_embedding_client_config_rejects_non_positive_batch_size(value: int) -> None:
    with pytest.raises(ValidationError):
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com", batch_size=value)


def test_http_embedding_client_config_default_headers_not_shared() -> None:
    first = HTTPEmbeddingClientConfig(endpoint_url="https://example.com/a")
    second = HTTPEmbeddingClientConfig(endpoint_url="https://example.com/b")
    first.headers["X-Test"] = "a"
    second.headers["X-Test"] = "b"
    assert first.headers == {"X-Test": "a"}
    assert second.headers == {"X-Test": "b"}


def test_http_embedding_client_supports_response_format_a() -> None:
    transport = RecordingTransport(
        [(200, b'{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}')]
    )
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    vectors = client.embed_texts(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_http_embedding_client_supports_response_format_b() -> None:
    transport = RecordingTransport(
        [(200, b'{"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}')]
    )
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    vectors = client.embed_texts(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_http_embedding_client_batches_requests() -> None:
    transport = RecordingTransport(
        [
            (200, b'{"embeddings": [[0.1], [0.2]]}'),
            (200, b'{"embeddings": [[0.3], [0.4]]}'),
            (200, b'{"embeddings": [[0.5]]}'),
        ]
    )
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(
            endpoint_url="https://example.com/embed",
            batch_size=2,
        ),
        transport=transport,
    )
    vectors = client.embed_texts(["a", "b", "c", "d", "e"])
    assert len(transport.calls) == 3
    assert vectors == [[0.1], [0.2], [0.3], [0.4], [0.5]]


def test_http_embedding_client_includes_model_when_configured() -> None:
    transport = RecordingTransport([(200, b'{"embeddings": [[0.1]]}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(
            endpoint_url="https://example.com/embed",
            model_name="text-embedding-3-large",
        ),
        transport=transport,
    )
    client.embed_texts(["a"])
    assert transport.calls[0]["payload"]["model"] == "text-embedding-3-large"


def test_http_embedding_client_omits_model_when_not_configured() -> None:
    transport = RecordingTransport([(200, b'{"embeddings": [[0.1]]}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    client.embed_texts(["a"])
    assert "model" not in transport.calls[0]["payload"]


def test_http_embedding_client_raises_for_http_error_status() -> None:
    transport = RecordingTransport([(500, b'{"error": "boom"}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="HTTP 500"):
        client.embed_texts(["a"])


def test_http_embedding_client_raises_for_invalid_json() -> None:
    transport = RecordingTransport([(200, b"{invalid-json}")])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="invalid JSON"):
        client.embed_texts(["a"])


def test_http_embedding_client_raises_for_count_mismatch() -> None:
    transport = RecordingTransport([(200, b'{"embeddings": [[0.1]]}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="different number of vectors"):
        client.embed_texts(["a", "b"])


def test_http_embedding_client_raises_for_empty_vector() -> None:
    transport = RecordingTransport([(200, b'{"embeddings": [[]]}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="empty vector"):
        client.embed_texts(["a"])


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_http_embedding_client_raises_for_non_finite_values(value: float) -> None:
    body = json.dumps({"embeddings": [[0.1, value]]}, allow_nan=True).encode("utf-8")
    transport = RecordingTransport([(200, body)])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="non-finite values"):
        client.embed_texts(["a"])


def test_http_embedding_client_raises_for_non_numeric_values() -> None:
    transport = RecordingTransport([(200, b'{"embeddings": [[0.1, "bad"]]}')])
    client = HTTPEmbeddingClient(
        HTTPEmbeddingClientConfig(endpoint_url="https://example.com/embed"),
        transport=transport,
    )
    with pytest.raises(EmbeddingClientError, match="non-numeric values"):
        client.embed_texts(["a"])


def test_http_embedding_client_from_env_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_ENDPOINT_URL", "https://example.com/embed")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret-key")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")

    client = http_embedding_client_from_env()

    assert client._config.endpoint_url == "https://example.com/embed"
    assert client._config.model_name == "text-embedding-3-large"
    assert client._config.timeout_seconds == 12.5
    assert client._config.batch_size == 8
    assert client._config.headers["Authorization"] == "Bearer secret-key"


def test_http_embedding_client_from_env_supports_custom_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_EMBEDDING_ENDPOINT_URL", "https://example.com/embed")

    client = http_embedding_client_from_env("MY_EMBEDDING")

    assert client._config.endpoint_url == "https://example.com/embed"


def test_http_embedding_client_from_env_raises_when_endpoint_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_ENDPOINT_URL", raising=False)
    with pytest.raises(EmbeddingClientError, match="EMBEDDING_ENDPOINT_URL"):
        http_embedding_client_from_env()


def test_http_embedding_client_from_env_raises_for_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_ENDPOINT_URL", "https://example.com/embed")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "bad-timeout")
    with pytest.raises(EmbeddingClientError, match="EMBEDDING_TIMEOUT_SECONDS"):
        http_embedding_client_from_env()


def test_http_embedding_client_from_env_raises_for_invalid_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_ENDPOINT_URL", "https://example.com/embed")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "bad-batch")
    with pytest.raises(EmbeddingClientError, match="EMBEDDING_BATCH_SIZE"):
        http_embedding_client_from_env()
