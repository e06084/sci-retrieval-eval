"""Connectivity checks for benchmark runtime configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import ssl
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse, request

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    elapsed_ms: float


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")
    return _expand_env(raw)


def _redacted_endpoint(url: str, index: int) -> str:
    parsed = parse.urlparse(url)
    host = parsed.hostname or "unknown-host"
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    return f"endpoint[{index}] {parsed.scheme}://{host}{port}{path}"


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    basic_auth: tuple[str, str] | None = None,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    final_headers = {"Accept": "application/json"}
    if payload is not None:
        final_headers["Content-Type"] = "application/json"
    if headers:
        final_headers.update(headers)
    if basic_auth is not None:
        import base64

        token = base64.b64encode(f"{basic_auth[0]}:{basic_auth[1]}".encode())
        final_headers["Authorization"] = f"Basic {token.decode('ascii')}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=body, headers=final_headers, method=method)
    context = ssl.create_default_context()
    with request.urlopen(req, timeout=timeout, context=context) as response:
        text = response.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed = text[:500]
        return response.status, parsed


def _timed(name: str, fn: Any) -> CheckResult:
    start = time.monotonic()
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=str(detail), elapsed_ms=_elapsed_ms(start))
    except Exception as exc:
        return CheckResult(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            elapsed_ms=_elapsed_ms(start),
        )


def _elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)


def _check_s3(config: dict[str, Any]) -> CheckResult:
    def run() -> str:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is not installed") from exc

        section = config.get("s3") or {}
        client = boto3.client(
            "s3",
            endpoint_url=str(section.get("endpoint") or ""),
            aws_access_key_id=str(section.get("access_key_id") or ""),
            aws_secret_access_key=str(section.get("secret_access_key") or ""),
        )
        bucket = str(section.get("bucket") or "")
        prefix = str(section.get("prefix") or "")
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        count = len(response.get("Contents", []))
        return f"listed bucket={bucket!r} prefix={prefix!r} sample_keys={count}"

    return _timed("s3", run)


def _check_elasticsearch(config: dict[str, Any]) -> CheckResult:
    def run() -> str:
        section = config.get("elasticsearch") or {}
        base_url = str(section.get("url") or "").rstrip("/")
        username = str(section.get("username") or "")
        password = str(section.get("password") or "")
        status, payload = _http_json(
            f"{base_url}/",
            timeout=10,
            basic_auth=(username, password) if username or password else None,
        )
        version = ""
        if isinstance(payload, dict):
            version = str((payload.get("version") or {}).get("number") or "")
        return f"HTTP {status} version={version or 'unknown'}"

    return _timed("elasticsearch", run)


def _parse_tcp_address(address: str) -> tuple[str, int]:
    parsed = parse.urlparse(address)
    if parsed.scheme and parsed.hostname and parsed.port:
        return parsed.hostname, int(parsed.port)
    if ":" in address:
        host, port = address.rsplit(":", 1)
        return host, int(port)
    raise ValueError(f"unsupported address: {address!r}")


def _check_milvus(config: dict[str, Any]) -> CheckResult:
    def run() -> str:
        section = config.get("milvus") or {}
        address = str(section.get("address") or "")
        host, port = _parse_tcp_address(address)
        with socket.create_connection((host, port), timeout=10):
            pass
        try:
            from pymilvus import connections, utility
        except ImportError:
            return f"tcp ok {host}:{port}; pymilvus not installed"

        alias = f"connectivity_{uuid.uuid4().hex}"
        kwargs: dict[str, Any] = {
            "alias": alias,
            "host": host,
            "port": str(port),
        }
        username = str(section.get("username") or "")
        password = str(section.get("password") or "")
        db_name = str(section.get("db_name") or "")
        if username:
            kwargs["user"] = username
        if password:
            kwargs["password"] = password
        if db_name:
            kwargs["db_name"] = db_name
        try:
            connections.connect(**kwargs)
            collections = utility.list_collections(using=alias)
            return f"connected db={db_name or 'default'} collections={len(collections)}"
        finally:
            connections.disconnect(alias)

    return _timed("milvus", run)


def _endpoint_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_embedding_vector(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
            return [float(item) for item in payload["embeddings"][0]]
        rows = payload.get("data")
        if isinstance(rows, list) and rows:
            row = rows[0]
            if isinstance(row, dict) and isinstance(row.get("embedding"), list):
                return [float(item) for item in row["embedding"]]
    raise ValueError("embedding response does not contain a vector")


def _check_embedding_endpoints(config: dict[str, Any]) -> list[CheckResult]:
    section = config.get("embedding") or {}
    endpoints = section.get("endpoints") or []
    model = str(section.get("model") or "")
    dim = int(section.get("dim") or 0)
    timeout = float(section.get("timeout_sec") or 30)
    results: list[CheckResult] = []
    for index, endpoint in enumerate(endpoints):
        endpoint = endpoint or {}
        url = str(endpoint.get("url") or "")
        api_key = str(endpoint.get("api_key") or "")
        name = f"embedding.{index}"

        def run(url: str = url, api_key: str = api_key, index: int = index) -> str:
            status, payload = _http_json(
                url,
                method="POST",
                timeout=timeout,
                headers=_endpoint_headers(api_key),
                payload={"model": model, "input": ["connectivity probe"]},
            )
            vector = _extract_embedding_vector(payload)
            if dim and len(vector) != dim:
                raise ValueError(f"dimension mismatch got={len(vector)} expected={dim}")
            if any(not math.isfinite(item) for item in vector):
                raise ValueError("vector contains non-finite values")
            return f"{_redacted_endpoint(url, index)} HTTP {status} dim={len(vector)}"

        results.append(_timed(name, run))
    return results


def _parse_rerank_payload(payload: Any) -> int:
    if isinstance(payload, dict):
        rows = payload.get("results") or payload.get("data") or []
        if isinstance(rows, list):
            return len(rows)
    raise ValueError("rerank response does not contain results")


def _check_rerank_endpoints(config: dict[str, Any]) -> list[CheckResult]:
    section = config.get("rerank") or {}
    endpoints = section.get("endpoints") or []
    model = str(section.get("model") or "")
    timeout = float(section.get("timeout_sec") or 30)
    results: list[CheckResult] = []
    for index, endpoint in enumerate(endpoints):
        endpoint = endpoint or {}
        url = str(endpoint.get("url") or "")
        api_key = str(endpoint.get("api_key") or "")
        name = f"rerank.{index}"

        def run(url: str = url, api_key: str = api_key, index: int = index) -> str:
            status, payload = _http_json(
                url,
                method="POST",
                timeout=timeout,
                headers=_endpoint_headers(api_key),
                payload={
                    "model": model,
                    "query": "connectivity probe",
                    "documents": ["first document", "second document"],
                    "top_n": 2,
                    "return_documents": False,
                },
            )
            rows = _parse_rerank_payload(payload)
            return f"{_redacted_endpoint(url, index)} HTTP {status} results={rows}"

        results.append(_timed(name, run))
    return results


def _check_rewrite(config: dict[str, Any]) -> CheckResult | None:
    section = ((config.get("search_runtime") or {}).get("rewrite") or {})
    if not section or not section.get("enabled"):
        return None

    def run() -> str:
        base_url = str(section.get("base_url") or "").rstrip("/")
        api_key = str(section.get("api_key") or "")
        timeout = float(section.get("timeout_sec") or 30)
        status, payload = _http_json(
            f"{base_url}/chat/completions",
            method="POST",
            timeout=timeout,
            headers=_endpoint_headers(api_key),
            payload={
                "model": str(section.get("model") or ""),
                "messages": [
                    {"role": "system", "content": "Return one short search query."},
                    {"role": "user", "content": "connectivity probe"},
                ],
                "temperature": float(section.get("temperature") or 0),
                "max_tokens": int(section.get("max_tokens") or 64),
            },
        )
        choices = payload.get("choices") if isinstance(payload, dict) else None
        return f"HTTP {status} choices={len(choices or [])}"

    return _timed("rewrite", run)


def _check_go_api(config: dict[str, Any]) -> CheckResult | None:
    section = config.get("search_runtime") or {}
    api_url = str(section.get("go_api_url") or "").rstrip("/")
    if not api_url:
        return None

    def run() -> str:
        status, payload = _http_json(f"{api_url}/healthz", timeout=5)
        if isinstance(payload, dict):
            keys = ",".join(sorted(str(key) for key in payload.keys()))
        else:
            keys = "non-json"
        return f"HTTP {status} keys={keys}"

    return _timed("go_api", run)


def _as_dict(result: CheckResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "ok": result.ok,
        "detail": result.detail,
        "elapsed_ms": result.elapsed_ms,
    }


def run_checks(config_path: Path) -> list[CheckResult]:
    config = _load_config(config_path)
    results: list[CheckResult] = [
        _check_s3(config),
        _check_elasticsearch(config),
        _check_milvus(config),
    ]
    results.extend(_check_embedding_endpoints(config))
    results.extend(_check_rerank_endpoints(config))
    rewrite = _check_rewrite(config)
    if rewrite is not None:
        results.append(rewrite)
    go_api = _check_go_api(config)
    if go_api is not None:
        results.append(go_api)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check connectivity for benchmark config.yaml")
    parser.add_argument(
        "--config",
        default=".local_artifacts/configs/sciverse_benchmark_config.yaml",
        help="Path to copied benchmark config.yaml",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    results = run_checks(Path(args.config))
    if args.json:
        print(json.dumps([_as_dict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        for result in results:
            status = "OK" if result.ok else "FAIL"
            print(f"[{status}] {result.name}: {result.detail} ({result.elapsed_ms} ms)")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
