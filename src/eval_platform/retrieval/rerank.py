"""HTTP rerank adapter."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.config import RerankConfig
from eval_platform.retrieval.clients import RerankClient
from eval_platform.retrieval.schema import RetrievalHit


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RerankAdapterError(Exception):
    """Raised when a rerank adapter cannot produce valid reranked hits."""


class HTTPRerankClientConfig(BaseModel):
    """Configuration for the HTTP rerank client."""

    endpoint_url: str
    endpoint_id: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=0, ge=0)

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_id", "model_name")
    @classmethod
    def validate_optional_strings(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")


class RerankConsistencyCheckResult(BaseModel):
    """Result of comparing rerank order across endpoints."""

    endpoint_ids: list[str]
    passed: bool
    query: str
    document_count: int
    rankings: dict[str, list[int]]
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint_ids")
    @classmethod
    def validate_endpoint_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("endpoint_ids must not be empty")
        for endpoint_id in value:
            if not endpoint_id.strip():
                raise ValueError("endpoint_ids must not contain empty values")
        return value

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_failure_reason(self) -> RerankConsistencyCheckResult:
        if self.passed and self.failure_reason is not None:
            raise ValueError("failure_reason must be None when passed=True")
        if not self.passed and not self.failure_reason:
            raise ValueError("failure_reason is required when passed=False")
        return self


def _default_transport(
    endpoint_url: str,
    payload: bytes,
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint_url,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise RerankAdapterError("Rerank endpoint request failed") from exc


class HTTPRerankClient:
    """HTTP rerank client using an injectable transport for tests."""

    def __init__(
        self,
        config: HTTPRerankClientConfig,
        *,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or _default_transport

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Return hits reranked by the configured HTTP endpoint."""

        if not hits or top_n <= 0:
            return []

        request_top_n = len(hits)
        body: dict[str, Any] = {
            "query": query,
            "documents": [hit.text if hit.text else " " for hit in hits],
            "top_n": request_top_n,
            "return_documents": False,
        }
        if self._config.model_name is not None:
            body["model"] = self._config.model_name

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        response_body = self._send(payload, headers)
        scored_indexes = _parse_rerank_response(response_body, candidate_count=len(hits))
        if not scored_indexes:
            raise RerankAdapterError("Rerank endpoint response did not contain usable results")

        scored_indexes.sort(key=lambda item: (-item[1], item[0]))
        return [
            hits[index].model_copy(update={"score": score})
            for index, score in scored_indexes[:top_n]
        ]

    def _send(self, payload: bytes, headers: dict[str, str]) -> bytes:
        attempts = self._config.max_retries + 1
        last_error: RerankAdapterError | None = None
        for _ in range(attempts):
            try:
                result = self._transport(
                    self._config.endpoint_url,
                    payload,
                    headers,
                    self._config.timeout_seconds,
                )
                status_code, response_body = _validate_transport_result(result)
                if 200 <= status_code < 300:
                    return response_body
                last_error = RerankAdapterError(
                    f"Rerank endpoint returned HTTP {status_code}"
                )
            except RerankAdapterError as exc:
                last_error = exc
            except Exception as exc:
                last_error = RerankAdapterError("Rerank endpoint request failed")
                last_error.__cause__ = exc
        if last_error is not None:
            raise last_error
        raise RerankAdapterError("Rerank endpoint request failed")


def rerank_client_from_config(
    config: RerankConfig,
    *,
    endpoint_index: int = 0,
) -> HTTPRerankClient:
    """Create a fixed-endpoint HTTP rerank client from platform config."""

    if not config.endpoints:
        raise RerankAdapterError("rerank.endpoints must not be empty")
    if endpoint_index < 0 or endpoint_index >= len(config.endpoints):
        raise RerankAdapterError("endpoint_index is out of range")

    endpoint = config.endpoints[endpoint_index]
    if endpoint.url is None or not endpoint.url.strip():
        raise RerankAdapterError("rerank endpoint url is required")

    return HTTPRerankClient(
        HTTPRerankClientConfig(
            endpoint_url=endpoint.url,
            endpoint_id=f"endpoint-{endpoint_index}",
            model_name=config.model,
            api_key=endpoint.api_key,
            timeout_seconds=60.0 if config.timeout_sec is None else config.timeout_sec,
            max_retries=0 if config.max_retries is None else config.max_retries,
        )
    )


def run_rerank_consistency_check(
    clients: Sequence[RerankClient],
    *,
    endpoint_ids: Sequence[str],
    query: str,
    documents: Sequence[str],
    top_n: int | None = None,
) -> RerankConsistencyCheckResult:
    """Compare rerank output order across already-constructed clients."""

    if len(clients) != len(endpoint_ids):
        raise RerankAdapterError("clients length must match endpoint_ids length")
    if not endpoint_ids:
        raise RerankAdapterError("endpoint_ids must not be empty")
    if not query.strip():
        raise RerankAdapterError("query must not be empty")

    endpoint_id_list = [str(endpoint_id) for endpoint_id in endpoint_ids]
    synthetic_hits = [
        RetrievalHit(
            chunk_id=str(index),
            doc_id=f"doc-{index}",
            text=document,
            score=0.0,
            recall_source="consistency_check",
        )
        for index, document in enumerate(documents)
    ]
    effective_top_n = top_n if top_n is not None else len(synthetic_hits)
    rankings: dict[str, list[int]] = {}

    for endpoint_id, client in zip(endpoint_id_list, clients, strict=True):
        try:
            reranked = client.rerank(query, synthetic_hits, effective_top_n)
            rankings[endpoint_id] = _ranking_indexes(reranked, document_count=len(documents))
        except Exception as exc:
            return RerankConsistencyCheckResult(
                endpoint_ids=endpoint_id_list,
                passed=False,
                query=query,
                document_count=len(documents),
                rankings=rankings,
                failure_reason=(
                    f"Endpoint {endpoint_id} failed: {type(exc).__name__}: {exc}"
                ),
            )

    reference = next(iter(rankings.values()), [])
    for endpoint_id, ranking in rankings.items():
        if ranking != reference:
            return RerankConsistencyCheckResult(
                endpoint_ids=endpoint_id_list,
                passed=False,
                query=query,
                document_count=len(documents),
                rankings=rankings,
                failure_reason=f"Endpoint {endpoint_id} ranking differs from reference",
            )

    return RerankConsistencyCheckResult(
        endpoint_ids=endpoint_id_list,
        passed=True,
        query=query,
        document_count=len(documents),
        rankings=rankings,
    )


def _validate_transport_result(result: Any) -> tuple[int, bytes]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RerankAdapterError("Rerank transport returned an invalid response")
    status_code, response_body = result
    if not isinstance(status_code, int):
        raise RerankAdapterError("Rerank transport returned an invalid status code")
    if not isinstance(response_body, bytes):
        raise RerankAdapterError("Rerank transport returned a non-bytes response body")
    return status_code, response_body


def _parse_rerank_response(
    response_body: bytes,
    *,
    candidate_count: int,
) -> list[tuple[int, float]]:
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RerankAdapterError("Rerank endpoint returned invalid JSON") from exc

    rows: list[Any] | None = None
    format_name: str | None = None
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            rows = payload["results"]
            format_name = "results"
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
            format_name = "data"

    if rows is None or format_name is None:
        raise RerankAdapterError("Rerank endpoint response format is not supported")

    seen: set[int] = set()
    scored: list[tuple[int, float]] = []
    for row in rows:
        parsed = _parse_rerank_row(row, format_name=format_name)
        if parsed is None:
            continue
        index, score = parsed
        if index in seen or index < 0 or index >= candidate_count:
            continue
        seen.add(index)
        scored.append((index, score))
    return scored


def _parse_rerank_row(row: Any, *, format_name: str) -> tuple[int, float] | None:
    if not isinstance(row, dict):
        return None
    index_key = "index" if format_name == "results" else "document_index"
    score_key = "relevance_score" if format_name == "results" else "score"
    raw_index = row.get(index_key)
    raw_score = row.get(score_key)
    if not isinstance(raw_index, int) or isinstance(raw_index, bool):
        return None
    if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
        return None
    score = float(raw_score)
    if not math.isfinite(score):
        return None
    return raw_index, score


def _ranking_indexes(
    hits: Sequence[RetrievalHit],
    *,
    document_count: int,
) -> list[int]:
    indexes: list[int] = []
    for hit in hits:
        try:
            index = int(hit.chunk_id)
        except ValueError as exc:
            raise RerankAdapterError(
                f"Rerank client returned unknown synthetic chunk_id: {hit.chunk_id}"
            ) from exc
        if index < 0 or index >= document_count:
            raise RerankAdapterError(
                f"Rerank client returned out-of-range synthetic chunk_id: {hit.chunk_id}"
            )
        indexes.append(index)
    return indexes
