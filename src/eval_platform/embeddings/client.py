"""Embedding client protocol and fake implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable
from urllib import error, request

from pydantic import BaseModel, Field, ValidationInfo, field_validator


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class EmbeddingClientError(Exception):
    """Raised when an embedding client cannot produce valid vectors."""


class HTTPEmbeddingClientConfig(BaseModel):
    """Configuration for the HTTP embedding client."""

    endpoint_url: str
    model_name: str | None = None
    timeout_seconds: float = 60.0
    headers: dict[str, str] = Field(default_factory=dict)
    batch_size: int = Field(default=32, gt=0)
    response_vector_field: str = "embedding"

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value

    @field_validator("response_vector_field")
    @classmethod
    def validate_response_vector_field(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


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


def _default_transport(
    endpoint_url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, bytes]:
    req = request.Request(
        endpoint_url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return response.status, response.read()
    except error.HTTPError as exc:
        _ = exc.read()
        raise EmbeddingClientError(f"Embedding endpoint returned HTTP {exc.code}") from exc
    except error.URLError as exc:
        raise EmbeddingClientError("Embedding endpoint request failed") from exc


class HTTPEmbeddingClient:
    """HTTP embedding client using an injectable transport for tests."""

    def __init__(
        self,
        config: HTTPEmbeddingClientConfig,
        *,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or _default_transport

    def _build_payload(self, texts: Sequence[str]) -> bytes:
        payload: dict[str, Any] = {"texts": list(texts)}
        if self._config.model_name is not None:
            payload["model"] = self._config.model_name
        return json.dumps(payload).encode("utf-8")

    def _extract_vectors(self, response_body: bytes, expected_count: int) -> list[list[float]]:
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingClientError("Embedding endpoint returned invalid JSON") from exc

        vectors: list[Any] | None = None
        if isinstance(payload, dict):
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list):
                vectors = embeddings
            else:
                data = payload.get("data")
                if isinstance(data, list):
                    vectors = [row.get(self._config.response_vector_field) for row in data]

        if vectors is None:
            raise EmbeddingClientError("Embedding endpoint response format is not supported")
        if len(vectors) != expected_count:
            raise EmbeddingClientError("Embedding endpoint returned a different number of vectors")

        normalized: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingClientError("Embedding endpoint returned an empty vector")
            converted: list[float] = []
            for item in vector:
                if not isinstance(item, (int, float)):
                    raise EmbeddingClientError(
                        "Embedding endpoint returned a vector with non-numeric values"
                    )
                value = float(item)
                if not math.isfinite(value):
                    raise EmbeddingClientError(
                        "Embedding endpoint returned a vector with non-finite values"
                    )
                converted.append(value)
            normalized.append(converted)
        return normalized

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}
        headers.update(self._config.headers)

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start : start + self._config.batch_size]
            payload = self._build_payload(batch)
            result = self._transport(
                self._config.endpoint_url,
                payload,
                headers,
                self._config.timeout_seconds,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise EmbeddingClientError("Embedding transport returned an invalid response")
            status_code, response_body = result
            if not isinstance(status_code, int):
                raise EmbeddingClientError("Embedding transport returned an invalid status code")
            if status_code < 200 or status_code >= 300:
                raise EmbeddingClientError(f"Embedding endpoint returned HTTP {status_code}")
            if not isinstance(response_body, bytes):
                raise EmbeddingClientError("Embedding transport returned a non-bytes response body")
            all_vectors.extend(self._extract_vectors(response_body, len(batch)))
        return all_vectors


def http_embedding_client_from_env(prefix: str = "EMBEDDING") -> HTTPEmbeddingClient:
    """Create an HTTP embedding client from environment variables."""
    normalized_prefix = prefix.strip()
    if not normalized_prefix:
        raise EmbeddingClientError("Environment variable prefix must not be empty")

    endpoint_key = f"{normalized_prefix}_ENDPOINT_URL"
    model_key = f"{normalized_prefix}_MODEL_NAME"
    api_key_key = f"{normalized_prefix}_API_KEY"
    timeout_key = f"{normalized_prefix}_TIMEOUT_SECONDS"
    batch_key = f"{normalized_prefix}_BATCH_SIZE"

    endpoint_url = os.environ.get(endpoint_key)
    if endpoint_url is None or not endpoint_url.strip():
        raise EmbeddingClientError(f"Missing required environment variable: {endpoint_key}")

    model_name = os.environ.get(model_key)
    timeout_seconds = 60.0
    batch_size = 32

    raw_timeout = os.environ.get(timeout_key)
    if raw_timeout is not None:
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as exc:
            raise EmbeddingClientError(
                f"Invalid float value for {timeout_key}: {raw_timeout!r}"
            ) from exc

    raw_batch_size = os.environ.get(batch_key)
    if raw_batch_size is not None:
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise EmbeddingClientError(
                f"Invalid integer value for {batch_key}: {raw_batch_size!r}"
            ) from exc

    headers: dict[str, str] = {}
    api_key = os.environ.get(api_key_key)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    config = HTTPEmbeddingClientConfig(
        endpoint_url=endpoint_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        headers=headers,
        batch_size=batch_size,
    )
    return HTTPEmbeddingClient(config)
