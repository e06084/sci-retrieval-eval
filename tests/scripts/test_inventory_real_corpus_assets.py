"""Tests for real corpus asset inventory helpers."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval_platform.artifacts import ArtifactFile, ArtifactManifest, LocalArtifactStore

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from corpus_asset_common import (  # noqa: E402
    DATASETS_BY_NAME,
    artifact_ids_for_dataset,
    inventory_corpus_assets,
    safe_json_dumps,
)


class FakeS3Client:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.put_calls: list[dict[str, Any]] = []

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        MaxKeys: int | None = None,
    ) -> dict[str, object]:
        matches = [{"Key": key} for key in self.keys if key.startswith(Prefix)]
        if MaxKeys is not None:
            matches = matches[:MaxKeys]
        return {"Contents": matches, "IsTruncated": False}

    def put_object(self, **kwargs: Any) -> None:
        self.put_calls.append(kwargs)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


def _manifest(
    artifact_type: str,
    artifact_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_at=datetime(2026, 5, 27, tzinfo=UTC),
        files=[ArtifactFile(path="_placeholder", size_bytes=0)],
        metadata=metadata or {},
    )


def test_dataset_slug_and_artifact_id_mapping() -> None:
    spec = DATASETS_BY_NAME["IFIRNFCorpus"]

    assert spec.slug == "ifir_nfcorpus"
    assert artifact_ids_for_dataset(spec, "run_001") == {
        "raw_dataset": "ifir_nfcorpus_run_001_raw",
        "normalized_dataset": "ifir_nfcorpus_run_001_normalized",
        "chunked_corpus": "ifir_nfcorpus_run_001_chunks",
        "embeddings": "ifir_nfcorpus_run_001_embeddings",
        "elasticsearch_index": "ifir_nfcorpus_run_001_es_index",
        "milvus_collection": "ifir_nfcorpus_run_001_milvus_collection",
    }


def test_inventory_identifies_complete_artifact_and_extracts_metadata(
    store: LocalArtifactStore,
) -> None:
    spec = DATASETS_BY_NAME["IFIRNFCorpus"]
    artifact_id = "ifir_nfcorpus_run_001_normalized"
    store.write_manifest(
        "normalized_dataset",
        artifact_id,
        _manifest(
            "normalized_dataset",
            artifact_id,
            metadata={
                "task_name": "IFIRNFCorpus",
                "corpus_count": 10,
                "query_count": 2,
                "qrel_count": 3,
                "password": "secret",
            },
        ),
    )
    store.mark_success("normalized_dataset", artifact_id)

    inventory = inventory_corpus_assets(
        store=store,
        raw_client=FakeS3Client(["sciverse_benchmark/raw/ifir_nfcorpus/corpus.jsonl"]),
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        datasets=[spec],
    )

    dataset = inventory["datasets"]["IFIRNFCorpus"]
    assert dataset["raw_prefix_exists"] is True
    records = dataset["artifacts"]["normalized_dataset"]
    assert records[0]["artifact_id"] == artifact_id
    assert records[0]["complete"] is True
    assert records[0]["has_success"] is True
    assert records[0]["metadata_summary"] == {
        "corpus_count": 10,
        "query_count": 2,
        "qrel_count": 3,
        "task_name": "IFIRNFCorpus",
    }
    assert "normalized_dataset" not in dataset["missing"]


def test_inventory_identifies_missing_success(store: LocalArtifactStore) -> None:
    spec = DATASETS_BY_NAME["NFCorpus"]
    artifact_id = "nfcorpus_run_001_chunks"
    store.write_manifest(
        "chunked_corpus",
        artifact_id,
        _manifest(
            "chunked_corpus",
            artifact_id,
            metadata={"task_name": "NFCorpus", "chunk_count": 7},
        ),
    )

    inventory = inventory_corpus_assets(
        store=store,
        raw_client=FakeS3Client(["sciverse_benchmark/raw/nfcorpus/corpus.jsonl"]),
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        datasets=[spec],
    )

    records = inventory["datasets"]["NFCorpus"]["artifacts"]["chunked_corpus"]
    assert records[0]["complete"] is False
    assert records[0]["has_manifest"] is True
    assert records[0]["has_success"] is False
    assert "chunked_corpus" in inventory["datasets"]["NFCorpus"]["missing"]


def test_inventory_does_not_match_slug_substrings(store: LocalArtifactStore) -> None:
    store.write_manifest(
        "chunked_corpus",
        "ifir_nfcorpus_run_001_chunks",
        _manifest(
            "chunked_corpus",
            "ifir_nfcorpus_run_001_chunks",
            metadata={"task_name": "IFIRNFCorpus", "chunk_count": 7},
        ),
    )
    store.mark_success("chunked_corpus", "ifir_nfcorpus_run_001_chunks")

    inventory = inventory_corpus_assets(
        store=store,
        raw_client=FakeS3Client(["sciverse_benchmark/raw/nfcorpus/corpus.jsonl"]),
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        datasets=[DATASETS_BY_NAME["NFCorpus"]],
    )

    assert inventory["datasets"]["NFCorpus"]["artifacts"]["chunked_corpus"] == []
    assert "chunked_corpus" in inventory["datasets"]["NFCorpus"]["missing"]


def test_inventory_marks_raw_prefix_missing(store: LocalArtifactStore) -> None:
    inventory = inventory_corpus_assets(
        store=store,
        raw_client=FakeS3Client([]),
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        datasets=[DATASETS_BY_NAME["SciFact"]],
    )

    dataset = inventory["datasets"]["SciFact"]
    assert dataset["raw_prefix_exists"] is False
    assert dataset["missing"][0] == "raw_prefix"


def test_safe_json_dumps_redacts_secrets() -> None:
    payload = {
        "s3": {"access_key_id": "abc", "secret_access_key": "def"},
        "headers": {"Authorization": "Bearer token"},
        "nested": [{"api_key": "secret"}, {"ok": "value"}],
    }

    text = safe_json_dumps(payload)

    assert "abc" not in text
    assert "def" not in text
    assert "Bearer token" not in text
    assert '"***"' in text
