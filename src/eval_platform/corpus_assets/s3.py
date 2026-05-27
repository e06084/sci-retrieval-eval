"""S3, output, and redaction helpers for corpus asset scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_platform.artifacts import S3ArtifactStore
from eval_platform.config import PlatformConfig, dump_redacted_config, load_platform_config
from eval_platform.corpus_assets.registry import CorpusAssetError

_SENSITIVE_KEY_PARTS = (
    "access_key",
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
)


def redact_sensitive_values(value: Any) -> Any:
    """Return a copy of a JSON-like payload with sensitive keys redacted."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
                out[key] = "***"
            else:
                out[key] = redact_sensitive_values(item)
        return out
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    return value


def safe_json_dumps(payload: Any) -> str:
    """Serialize a payload after redacting sensitive keys."""

    return (
        json.dumps(
            redact_sensitive_values(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def make_s3_client(config: PlatformConfig) -> Any:
    """Create a boto3 S3 client from platform config."""

    try:
        import boto3
    except ImportError as exc:
        raise CorpusAssetError(
            "boto3 is required for real S3 inventory; install the s3 extra"
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=config.s3.endpoint,
        aws_access_key_id=config.s3.access_key_id,
        aws_secret_access_key=config.s3.secret_access_key,
    )


def make_s3_artifact_store(
    *,
    config: PlatformConfig,
    s3_prefix: str,
    client: Any,
) -> S3ArtifactStore:
    """Create the artifact store for a target S3 prefix."""

    if not config.s3.bucket:
        raise CorpusAssetError("config.s3.bucket is required")
    return S3ArtifactStore(
        bucket=config.s3.bucket,
        prefix=s3_prefix.strip("/"),
        client=client,
    )


def raw_prefix_exists(client: Any, *, bucket: str, prefix: str) -> bool:
    """Return whether a raw S3 prefix contains at least one object."""

    list_prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    response = client.list_objects_v2(Bucket=bucket, Prefix=list_prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def load_config_and_client(config_path: Path) -> tuple[PlatformConfig, Any]:
    """Load platform config and create an S3 client."""

    config = load_platform_config(config_path)
    return config, make_s3_client(config)


def output_payload(payload: dict[str, Any], output: Path | None) -> None:
    """Print JSON and optionally write it to a local file."""

    text = safe_json_dumps(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared config/S3 arguments."""

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s3-prefix", default="test_sciverse_benchmark")
    parser.add_argument("--raw-prefix", default="sciverse_benchmark/raw")
    parser.add_argument("--output", type=Path, default=None)


def redacted_config_summary(config: PlatformConfig) -> dict[str, Any]:
    """Return a safe config summary for reports."""

    return dump_redacted_config(config)
