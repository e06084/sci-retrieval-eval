"""Tests for artifact catalog records."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from eval_platform.artifacts import (
    ARTIFACT_CATALOG_ARTIFACT_TYPE,
    ArtifactManifest,
    LocalArtifactStore,
    build_artifact_catalog_record,
    find_catalog_record_by_fingerprint,
    read_artifact_catalog,
    sync_artifact_catalog_from_store,
    upsert_artifact_catalog_record,
)


def test_catalog_upsert_read_and_find_by_fingerprint(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write_manifest(
        store,
        "retrieval_run",
        "retrieval-a",
        {
            "asset_fingerprint_sha256": "fp-retrieval",
            "retrieval_mode": "es",
        },
    )

    record = build_artifact_catalog_record(store, "retrieval_run", "retrieval-a")
    upsert_artifact_catalog_record(store, record)
    records = read_artifact_catalog(store)
    found = find_catalog_record_by_fingerprint(
        records,
        artifact_type="retrieval_run",
        asset_fingerprint_sha256="fp-retrieval",
    )

    assert store.is_complete(ARTIFACT_CATALOG_ARTIFACT_TYPE, "default") is True
    assert len(records) == 1
    assert records[0].metadata_summary == {
        "asset_fingerprint_sha256": "fp-retrieval",
        "retrieval_mode": "es",
    }
    assert found is not None
    assert found.artifact_id == "retrieval-a"


def test_catalog_sync_scans_store_without_catalog_self_reference(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _write_manifest(
        store,
        "metrics_run",
        "metrics-a",
        {
            "asset_fingerprint_sha256": "fp-metrics",
            "main_score": 1.0,
        },
    )

    sync_artifact_catalog_from_store(store)
    records = read_artifact_catalog(store)

    assert [(record.artifact_type, record.artifact_id) for record in records] == [
        ("metrics_run", "metrics-a")
    ]


def _write_manifest(
    store: LocalArtifactStore,
    artifact_type: str,
    artifact_id: str,
    metadata: dict[str, object],
) -> None:
    store.write_manifest(
        artifact_type,
        artifact_id,
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            created_at=datetime.now(UTC),
            metadata=metadata,
        ),
    )
    store.mark_success(artifact_type, artifact_id)
