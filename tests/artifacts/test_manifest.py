"""Tests for artifact manifest schema."""

from datetime import UTC, datetime

from eval_platform.artifacts import ArtifactFile, ArtifactManifest


def test_manifest_defaults() -> None:
    manifest = ArtifactManifest(
        artifact_id="sample_001",
        artifact_type="normalized_dataset",
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
    )

    assert manifest.schema_version == "1"
    assert manifest.dependencies == []
    assert manifest.files == []
    assert manifest.metadata == {}


def test_manifest_json_round_trip() -> None:
    manifest = ArtifactManifest(
        artifact_id="sample_001",
        artifact_type="normalized_dataset",
        created_at=datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC),
        files=[ArtifactFile(path="corpus.jsonl", size_bytes=128, sha256="abc123")],
        metadata={"source": "test"},
    )

    payload = manifest.model_dump(mode="json")
    restored = ArtifactManifest.model_validate(payload)

    assert restored == manifest
