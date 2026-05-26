"""Retrieval run artifact read/write helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.retrieval.jsonl import (
    dump_retrieval_results_jsonl,
    load_retrieval_results_jsonl,
)
from eval_platform.retrieval.schema import RetrievalQueryResult

RETRIEVAL_RUN_ARTIFACT_TYPE = "retrieval_run"
RESULTS_DIR = "results"
_SYSTEM_METADATA_FIELDS = {
    "stage",
    "query_count",
    "succeeded_query_count",
    "failed_query_count",
    "queries_per_shard",
    "result_file_count",
    "result_record_count",
}


class RetrievalArtifactError(Exception):
    """Raised when retrieval artifact validation fails."""


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _result_shard_path(index: int) -> str:
    return f"{RESULTS_DIR}/part-{index:05d}.jsonl"


def write_retrieval_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    records: list[RetrievalQueryResult],
    *,
    queries_per_shard: int = 1000,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write a retrieval_run artifact with sharded JSONL result records."""

    if queries_per_shard <= 0:
        raise RetrievalArtifactError("queries_per_shard must be greater than 0")

    files: list[ArtifactFile] = []
    shard_count = max(1, (len(records) + queries_per_shard - 1) // queries_per_shard)
    for shard_index in range(shard_count):
        start = shard_index * queries_per_shard
        stop = start + queries_per_shard
        shard_records = records[start:stop]
        payload = dump_retrieval_results_jsonl(shard_records).encode("utf-8")
        path = _result_shard_path(shard_index)
        store.put_file(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id, path, payload)
        files.append(
            ArtifactFile(
                path=path,
                size_bytes=len(payload),
                sha256=_sha256_hexdigest(payload),
            )
        )

    succeeded_query_count = sum(1 for record in records if record.error is None)
    failed_query_count = len(records) - succeeded_query_count
    manifest_metadata = {
        key: value for key, value in (metadata or {}).items() if key not in _SYSTEM_METADATA_FIELDS
    }
    manifest_metadata.update(
        {
            "stage": "retrieval_run",
            "query_count": len(records),
            "succeeded_query_count": succeeded_query_count,
            "failed_query_count": failed_query_count,
            "queries_per_shard": queries_per_shard,
            "result_file_count": len(files),
            "result_record_count": len(records),
        }
    )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=files,
    )
    store.write_manifest(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_retrieval_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> list[RetrievalQueryResult]:
    """Read all retrieval result records from a retrieval_run artifact."""

    if require_complete and not store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {RETRIEVAL_RUN_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id)
    records: list[RetrievalQueryResult] = []
    for artifact_file in manifest.files:
        if not artifact_file.path.startswith(f"{RESULTS_DIR}/"):
            continue
        payload = store.get_file(
            RETRIEVAL_RUN_ARTIFACT_TYPE,
            artifact_id,
            artifact_file.path,
        ).decode("utf-8")
        records.extend(load_retrieval_results_jsonl(payload))
    return records
