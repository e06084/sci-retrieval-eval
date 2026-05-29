"""Helpers for writing asset fingerprints into artifact manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.metadata_keys import (
    METADATA_KEY_ASSET_FINGERPRINT,
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
)
from eval_platform.assets.fingerprint import build_asset_fingerprint

ASSET_FINGERPRINT_METADATA_KEYS = frozenset(
    {
        METADATA_KEY_ASSET_FINGERPRINT,
        METADATA_KEY_ASSET_FINGERPRINT_SHA256,
    }
)


class AssetFingerprintMetadataError(Exception):
    """Raised when a manifest does not carry a valid asset fingerprint."""


def asset_fingerprint_metadata(
    *,
    artifact_type: str,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    """Return manifest metadata entries for a stable logical asset fingerprint."""

    fingerprint = build_asset_fingerprint(
        artifact_type=artifact_type,
        components=components,
    )
    return {
        METADATA_KEY_ASSET_FINGERPRINT: fingerprint.model_dump(mode="json"),
        METADATA_KEY_ASSET_FINGERPRINT_SHA256: fingerprint.sha256,
    }


def add_asset_fingerprint_metadata(
    metadata: dict[str, Any],
    *,
    artifact_type: str,
    components: Mapping[str, Any] | None,
) -> None:
    """Mutate metadata in place by adding fingerprint fields when components exist."""

    if components is None:
        return
    metadata.update(
        asset_fingerprint_metadata(
            artifact_type=artifact_type,
            components=components,
        )
    )


def manifest_asset_fingerprint_sha256(manifest: ArtifactManifest) -> str | None:
    """Return a manifest's asset fingerprint sha, if present and non-empty."""

    value = manifest.metadata.get(METADATA_KEY_ASSET_FINGERPRINT_SHA256)
    if isinstance(value, str) and value.strip():
        return value
    return None


def require_manifest_asset_fingerprint_sha256(manifest: ArtifactManifest) -> str:
    """Return a manifest asset fingerprint or raise a clear error."""

    value = manifest_asset_fingerprint_sha256(manifest)
    if value is None:
        raise AssetFingerprintMetadataError(
            f"Manifest {manifest.artifact_type}/{manifest.artifact_id} "
            f"does not contain {METADATA_KEY_ASSET_FINGERPRINT_SHA256}"
        )
    return value


def strip_asset_fingerprint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata without system fingerprint fields."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in ASSET_FINGERPRINT_METADATA_KEYS
    }
