"""Lightweight artifact catalog records stored as JSONL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from eval_platform.artifacts.manifest import ArtifactFile, ArtifactManifest
from eval_platform.artifacts.metadata_keys import METADATA_KEY_ASSET_FINGERPRINT_SHA256
from eval_platform.artifacts.store import ArtifactNotFoundError, ArtifactStore
from eval_platform.artifacts.types import (
    ALL_ARTIFACT_TYPES,
    ARTIFACT_CATALOG_ARTIFACT_TYPE,
)
from eval_platform.assets import manifest_asset_fingerprint_sha256

CATALOG_RECORDS_FILENAME = "records.jsonl"
DEFAULT_CATALOG_ID = "default"
_SUMMARY_METADATA_KEYS = {
    "stage",
    "dataset",
    "dataset_name",
    "task_name",
    "dataset_key",
    "setting_key",
    "experiment_run_id",
    "suite_run_id",
    "source_normalized_dataset_artifact_id",
    "source_chunked_corpus_artifact_id",
    "source_embeddings_artifact_id",
    "raw_dataset_artifact_id",
    "chunked_corpus_artifact_id",
    "embeddings_artifact_id",
    "retrieval_run_artifact_id",
    "metrics_run_artifact_id",
    "index_name",
    "collection_name",
    "query_count",
    "qrel_count",
    "corpus_count",
    "chunk_count",
    "embedding_count",
    "indexed_count",
    "inserted_count",
    "main_score",
    "main_score_metric",
    "retrieval_mode",
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
}


class ArtifactCatalogRecord(BaseModel):
    """One searchable catalog entry for a materialized artifact."""

    artifact_type: str
    artifact_id: str
    artifact_uri: str
    complete: bool
    has_manifest: bool = True
    has_success: bool
    created_at: str | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    asset_fingerprint_sha256: str | None = None
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    metadata_summary: dict[str, Any] = Field(default_factory=dict)


def build_artifact_catalog_record(
    store: ArtifactStore,
    artifact_type: str,
    artifact_id: str,
) -> ArtifactCatalogRecord:
    """Build one catalog record from the current artifact manifest state."""

    manifest = store.read_manifest(artifact_type, artifact_id)
    complete = store.is_complete(artifact_type, artifact_id)
    return ArtifactCatalogRecord(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_uri=store.artifact_uri(artifact_type, artifact_id),
        complete=complete,
        has_manifest=True,
        has_success=complete,
        created_at=manifest.created_at.isoformat() if manifest.created_at else None,
        created_by=manifest.created_by,
        code_git_sha=manifest.code_git_sha,
        asset_fingerprint_sha256=manifest_asset_fingerprint_sha256(manifest),
        dependencies=[
            dependency.model_dump(mode="json") for dependency in manifest.dependencies
        ],
        metadata_summary=_metadata_summary(manifest),
    )


def scan_artifact_catalog_records(
    store: ArtifactStore,
    *,
    artifact_types: list[str] | None = None,
) -> list[ArtifactCatalogRecord]:
    """Scan an artifact store and return complete/incomplete catalog records."""

    selected_types = artifact_types or [
        artifact_type
        for artifact_type in ALL_ARTIFACT_TYPES
        if artifact_type != ARTIFACT_CATALOG_ARTIFACT_TYPE
    ]
    records: list[ArtifactCatalogRecord] = []
    for artifact_type in selected_types:
        for _current_type, artifact_id in store.list_artifacts(artifact_type):
            try:
                records.append(build_artifact_catalog_record(store, artifact_type, artifact_id))
            except Exception:
                records.append(
                    ArtifactCatalogRecord(
                        artifact_type=artifact_type,
                        artifact_id=artifact_id,
                        artifact_uri=store.artifact_uri(artifact_type, artifact_id),
                        complete=False,
                        has_manifest=False,
                        has_success=False,
                    )
                )
    return sorted(records, key=lambda item: (item.artifact_type, item.artifact_id))


def dump_artifact_catalog_jsonl(records: list[ArtifactCatalogRecord]) -> str:
    """Serialize catalog records as stable JSONL."""

    return "".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        + "\n"
        for record in records
    )


def load_artifact_catalog_jsonl(text: str) -> list[ArtifactCatalogRecord]:
    """Parse catalog JSONL text."""

    records: list[ArtifactCatalogRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        records.append(ArtifactCatalogRecord.model_validate_json(line))
    return records


def read_artifact_catalog(
    store: ArtifactStore,
    catalog_id: str = DEFAULT_CATALOG_ID,
) -> list[ArtifactCatalogRecord]:
    """Read catalog records from an artifact_catalog artifact."""

    try:
        payload = store.get_file(
            ARTIFACT_CATALOG_ARTIFACT_TYPE,
            catalog_id,
            CATALOG_RECORDS_FILENAME,
        )
    except ArtifactNotFoundError:
        return []
    return load_artifact_catalog_jsonl(payload.decode("utf-8"))


def write_artifact_catalog(
    store: ArtifactStore,
    records: list[ArtifactCatalogRecord],
    catalog_id: str = DEFAULT_CATALOG_ID,
    *,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write catalog records as an artifact_catalog artifact."""

    deduped = _dedupe_records(records)
    payload = dump_artifact_catalog_jsonl(deduped).encode("utf-8")
    store.put_file(
        ARTIFACT_CATALOG_ARTIFACT_TYPE,
        catalog_id,
        CATALOG_RECORDS_FILENAME,
        payload,
    )
    manifest = ArtifactManifest(
        artifact_id=catalog_id,
        artifact_type=ARTIFACT_CATALOG_ARTIFACT_TYPE,
        created_at=datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        metadata={
            "stage": ARTIFACT_CATALOG_ARTIFACT_TYPE,
            "record_count": len(deduped),
        },
        files=[ArtifactFile(path=CATALOG_RECORDS_FILENAME, size_bytes=len(payload))],
    )
    store.write_manifest(ARTIFACT_CATALOG_ARTIFACT_TYPE, catalog_id, manifest)
    store.mark_success(ARTIFACT_CATALOG_ARTIFACT_TYPE, catalog_id)
    return manifest


def upsert_artifact_catalog_record(
    store: ArtifactStore,
    record: ArtifactCatalogRecord,
    catalog_id: str = DEFAULT_CATALOG_ID,
    *,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Insert or replace one catalog record."""

    records = read_artifact_catalog(store, catalog_id)
    records = [
        existing
        for existing in records
        if (existing.artifact_type, existing.artifact_id)
        != (record.artifact_type, record.artifact_id)
    ]
    records.append(record)
    return write_artifact_catalog(
        store,
        records,
        catalog_id,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )


def sync_artifact_catalog_from_store(
    store: ArtifactStore,
    catalog_id: str = DEFAULT_CATALOG_ID,
    *,
    artifact_types: list[str] | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Rebuild/sync catalog records by scanning the artifact store."""

    records = scan_artifact_catalog_records(store, artifact_types=artifact_types)
    return write_artifact_catalog(
        store,
        records,
        catalog_id,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )


def find_catalog_record_by_fingerprint(
    records: list[ArtifactCatalogRecord],
    *,
    artifact_type: str,
    asset_fingerprint_sha256: str,
) -> ArtifactCatalogRecord | None:
    """Return the newest complete catalog record matching a fingerprint."""

    matches = [
        record
        for record in records
        if record.complete
        and record.artifact_type == artifact_type
        and record.asset_fingerprint_sha256 == asset_fingerprint_sha256
    ]
    if not matches:
        return None
    sorted_matches = sorted(
        matches,
        key=lambda item: (item.created_at or "", item.artifact_id),
        reverse=True,
    )
    return sorted_matches[0]


def _dedupe_records(records: list[ArtifactCatalogRecord]) -> list[ArtifactCatalogRecord]:
    by_key: dict[tuple[str, str], ArtifactCatalogRecord] = {}
    for record in records:
        by_key[(record.artifact_type, record.artifact_id)] = record
    return [by_key[key] for key in sorted(by_key)]


def _metadata_summary(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.metadata.items()
        if key in _SUMMARY_METADATA_KEYS and _is_summary_value(value)
    }


def _is_summary_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(item is None or isinstance(item, str | int | float | bool) for item in value)
    return False
