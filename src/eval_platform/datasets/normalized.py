"""Normalized dataset artifact read/write helpers."""

from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts.manifest import ArtifactDependency, ArtifactFile, ArtifactManifest
from eval_platform.artifacts.metadata_keys import (
    METADATA_KEY_ASSET_FINGERPRINT,
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
)
from eval_platform.artifacts.store import ArtifactIncompleteError, ArtifactStore
from eval_platform.artifacts.types import NORMALIZED_DATASET_ARTIFACT_TYPE
from eval_platform.assets import (
    add_asset_fingerprint_metadata,
    normalized_dataset_fingerprint_components,
)
from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl
from eval_platform.datasets.schema import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
)

CORPUS_FILENAME = "corpus.jsonl"
QUERIES_FILENAME = "queries.jsonl"
QRELS_FILENAME = "qrels.jsonl"
_SYSTEM_METADATA_FIELDS = {
    "corpus_count",
    "query_count",
    "qrel_count",
    METADATA_KEY_ASSET_FINGERPRINT,
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
}


def write_normalized_dataset_artifact(
    store: ArtifactStore,
    artifact_id: str,
    dataset: NormalizedDataset,
    *,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
) -> ArtifactManifest:
    """Write a normalized dataset artifact to the given store."""
    corpus_bytes = dump_jsonl(dataset.corpus).encode("utf-8")
    queries_bytes = dump_jsonl(dataset.queries).encode("utf-8")
    qrels_bytes = dump_jsonl(dataset.qrels).encode("utf-8")

    file_payloads = [
        (CORPUS_FILENAME, corpus_bytes),
        (QUERIES_FILENAME, queries_bytes),
        (QRELS_FILENAME, qrels_bytes),
    ]

    for filename, payload in file_payloads:
        store.put_file(NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id, filename, payload)

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(dataset.metadata)
    if metadata:
        manifest_metadata.update(metadata)
    manifest_metadata.update(
        {
            "corpus_count": len(dataset.corpus),
            "query_count": len(dataset.queries),
            "qrel_count": len(dataset.qrels),
        }
    )
    add_asset_fingerprint_metadata(
        manifest_metadata,
        artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
        components=_normalized_asset_fingerprint_components(manifest_metadata),
    )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=[
            ArtifactFile(path=filename, size_bytes=len(payload))
            for filename, payload in file_payloads
        ],
    )
    store.write_manifest(NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_normalized_dataset_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> NormalizedDataset:
    """Read a normalized dataset artifact from the given store."""
    if require_complete and not store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {NORMALIZED_DATASET_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id)
    dataset_metadata = {
        key: value
        for key, value in manifest.metadata.items()
        if key not in _SYSTEM_METADATA_FIELDS
    }

    corpus_text = store.get_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id, CORPUS_FILENAME
    ).decode("utf-8")
    queries_text = store.get_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id, QUERIES_FILENAME
    ).decode("utf-8")
    qrels_text = store.get_file(
        NORMALIZED_DATASET_ARTIFACT_TYPE, artifact_id, QRELS_FILENAME
    ).decode("utf-8")

    return NormalizedDataset(
        corpus=load_jsonl(corpus_text, CorpusRecord),
        queries=load_jsonl(queries_text, QueryRecord),
        qrels=load_jsonl(qrels_text, QrelRecord),
        metadata=dataset_metadata,
    )


def _normalized_asset_fingerprint_components(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    raw_fingerprint = metadata.get("raw_dataset_asset_fingerprint_sha256") or metadata.get(
        "raw_dataset_fingerprint"
    )
    normalizer_name = metadata.get("normalizer_name")
    schema_version = metadata.get("normalized_schema_version") or metadata.get("schema_version")
    if not (
        isinstance(raw_fingerprint, str)
        and raw_fingerprint.strip()
        and isinstance(normalizer_name, str)
        and normalizer_name.strip()
        and isinstance(schema_version, str)
        and schema_version.strip()
    ):
        return None

    normalizer_version = metadata.get("normalizer_version")
    if not isinstance(normalizer_version, str) or not normalizer_version.strip():
        normalizer_version = "1"

    normalizer_params = metadata.get("normalizer_params")
    if not isinstance(normalizer_params, dict):
        normalizer_params = {
            key: metadata[key]
            for key in ("split", "raw_format", "has_instructions")
            if key in metadata
        }

    return normalized_dataset_fingerprint_components(
        raw_dataset_fingerprint=raw_fingerprint,
        normalizer_name=normalizer_name,
        normalizer_version=normalizer_version,
        schema_version=schema_version,
        normalizer_params=normalizer_params,
    )
