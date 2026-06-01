#!/usr/bin/env python3
"""Inspect artifact manifests under an S3 artifact prefix.

This is a read-only diagnostic helper. It lists only artifact directory levels
with S3 delimiter queries, then reads each artifact's _MANIFEST.json and
_SUCCESS marker. It avoids scanning large chunk/embedding payload files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_platform.artifacts.metadata_keys import (  # noqa: E402
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
)
from eval_platform.artifacts.s3 import normalize_prefix  # noqa: E402
from eval_platform.artifacts.store import MANIFEST_FILENAME, SUCCESS_MARKER  # noqa: E402
from eval_platform.corpus_assets import (  # noqa: E402
    load_config_and_client,
    output_payload,
    redacted_config_summary,
)

_METADATA_SUMMARY_KEYS = (
    "stage",
    "dataset_key",
    "dataset_name",
    "task_name",
    "source_type",
    "source_uri",
    "raw_source_uri",
    "source_normalized_dataset_artifact_id",
    "source_chunked_corpus_artifact_id",
    "source_embeddings_artifact_id",
    "raw_dataset_artifact_id",
    "chunked_corpus_artifact_id",
    "embeddings_artifact_id",
    "elasticsearch_index_artifact_id",
    "milvus_collection_artifact_id",
    "index_name",
    "collection_name",
    "record_count",
    "doc_count",
    "corpus_count",
    "query_count",
    "qrel_count",
    "chunk_count",
    "embedding_count",
    "main_score",
    "main_score_metric",
    "retrieval_mode",
    "setting_name",
    "suite_run_id",
    "item_count",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Artifact prefix. Defaults to config.s3.prefix or sciverse_benchmark/assets.",
    )
    parser.add_argument(
        "--artifact-type",
        action="append",
        default=None,
        help="Artifact type to inspect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--artifact-id-contains",
        action="append",
        default=None,
        help="Only include artifact ids containing this substring. Can be repeated.",
    )
    parser.add_argument(
        "--fingerprint",
        choices=("any", "with", "without"),
        default="any",
        help="Filter by asset_fingerprint_sha256 presence.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum artifacts to return.")
    parser.add_argument("--include-manifest", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, client = load_config_and_client(args.config)
    if not config.s3.bucket:
        raise SystemExit("config.s3.bucket is required")

    s3_prefix = normalize_prefix(
        args.s3_prefix or config.s3.prefix or "sciverse_benchmark/assets"
    )
    artifact_types = args.artifact_type or _list_artifact_types(
        client,
        bucket=config.s3.bucket,
        s3_prefix=s3_prefix,
    )
    records: list[dict[str, Any]] = []

    for artifact_type in artifact_types:
        artifact_ids = _list_artifact_ids(
            client,
            bucket=config.s3.bucket,
            s3_prefix=s3_prefix,
            artifact_type=artifact_type,
        )
        for artifact_id in artifact_ids:
            if args.artifact_id_contains and not any(
                needle in artifact_id for needle in args.artifact_id_contains
            ):
                continue
            record = _inspect_artifact(
                client,
                bucket=config.s3.bucket,
                s3_prefix=s3_prefix,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                include_manifest=args.include_manifest,
            )
            if not _matches_fingerprint_filter(record, args.fingerprint):
                continue
            records.append(record)
            if len(records) >= args.limit:
                break
        if len(records) >= args.limit:
            break

    payload = {
        "kind": "s3_artifact_inventory_view",
        "s3_bucket": config.s3.bucket,
        "artifact_prefix": s3_prefix,
        "artifact_types": artifact_types,
        "filters": {
            "artifact_id_contains": args.artifact_id_contains or [],
            "fingerprint": args.fingerprint,
            "limit": args.limit,
        },
        "config": redacted_config_summary(config),
        "stats": _build_stats(records),
        "artifacts": records,
    }
    output_payload(payload, args.output)
    return payload


def _list_artifact_types(client: Any, *, bucket: str, s3_prefix: str) -> list[str]:
    return [
        prefix.rstrip("/").split("/")[-1]
        for prefix in _list_common_prefixes(client, bucket=bucket, prefix=s3_prefix)
    ]


def _list_artifact_ids(
    client: Any,
    *,
    bucket: str,
    s3_prefix: str,
    artifact_type: str,
) -> list[str]:
    prefix = "/".join(part for part in (s3_prefix, artifact_type) if part)
    return [
        artifact_prefix.rstrip("/").split("/")[-1]
        for artifact_prefix in _list_common_prefixes(client, bucket=bucket, prefix=prefix)
    ]


def _list_common_prefixes(client: Any, *, bucket: str, prefix: str) -> list[str]:
    list_prefix = f"{normalize_prefix(prefix)}/" if prefix else ""
    common_prefixes: list[str] = []
    continuation_token: str | None = None

    while True:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": list_prefix,
            "Delimiter": "/",
        }
        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**request)
        common_prefixes.extend(
            item["Prefix"] for item in response.get("CommonPrefixes", [])
        )
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    return sorted(common_prefixes)


def _inspect_artifact(
    client: Any,
    *,
    bucket: str,
    s3_prefix: str,
    artifact_type: str,
    artifact_id: str,
    include_manifest: bool,
) -> dict[str, Any]:
    artifact_prefix = "/".join(part for part in (s3_prefix, artifact_type, artifact_id) if part)
    manifest_key = f"{artifact_prefix}/{MANIFEST_FILENAME}"
    success_key = f"{artifact_prefix}/{SUCCESS_MARKER}"
    manifest = _read_json_object(client, bucket=bucket, key=manifest_key)
    has_success = _object_exists(client, bucket=bucket, key=success_key)
    metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
    fingerprint = metadata.get(METADATA_KEY_ASSET_FINGERPRINT_SHA256)

    record: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "artifact_uri": f"s3://{bucket}/{artifact_prefix}/",
        "complete": manifest is not None and has_success,
        "has_manifest": manifest is not None,
        "has_success": has_success,
        "has_asset_fingerprint": isinstance(fingerprint, str) and bool(fingerprint),
        METADATA_KEY_ASSET_FINGERPRINT_SHA256: fingerprint,
    }
    if isinstance(manifest, dict):
        record.update(
            {
                "created_at": manifest.get("created_at"),
                "created_by": manifest.get("created_by"),
                "code_git_sha": manifest.get("code_git_sha"),
                "dependency_count": len(manifest.get("dependencies", [])),
                "file_count": len(manifest.get("files", [])),
                "metadata_summary": _metadata_summary(metadata),
            }
        )
        if include_manifest:
            record["manifest"] = manifest
    return record


def _read_json_object(client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_not_found_error(exc):
            return None
        raise
    body = response["Body"]
    payload = body.read() if hasattr(body, "read") else body
    return json.loads(payload.decode("utf-8"))


def _object_exists(client: Any, *, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_not_found_error(exc):
            return False
        raise
    return True


def _is_not_found_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        return code in {"404", "NoSuchKey", "NotFound"}
    return False


def _metadata_summary(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in _METADATA_SUMMARY_KEYS
        if key in metadata and _is_summary_value(metadata[key])
    }


def _is_summary_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(item is None or isinstance(item, str | int | float | bool) for item in value)
    return False


def _matches_fingerprint_filter(record: dict[str, Any], fingerprint_filter: str) -> bool:
    if fingerprint_filter == "with":
        return bool(record.get("has_asset_fingerprint"))
    if fingerprint_filter == "without":
        return not bool(record.get("has_asset_fingerprint"))
    return True


def _build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    for record in records:
        artifact_type = str(record["artifact_type"])
        stats = by_type.setdefault(
            artifact_type,
            {
                "total": 0,
                "complete": 0,
                "with_asset_fingerprint": 0,
                "without_asset_fingerprint": 0,
            },
        )
        stats["total"] += 1
        if record.get("complete"):
            stats["complete"] += 1
        if record.get("has_asset_fingerprint"):
            stats["with_asset_fingerprint"] += 1
        else:
            stats["without_asset_fingerprint"] += 1
    return {
        "total": len(records),
        "complete": sum(1 for record in records if record.get("complete")),
        "with_asset_fingerprint": sum(
            1 for record in records if record.get("has_asset_fingerprint")
        ),
        "without_asset_fingerprint": sum(
            1 for record in records if not record.get("has_asset_fingerprint")
        ),
        "by_type": by_type,
    }


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
