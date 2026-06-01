"""Tests for the S3 artifact viewer script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import view_s3_artifacts as viewer_script  # noqa: E402


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3NotFound(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.list_requests: list[dict[str, Any]] = []

    def list_objects_v2(self, **request: Any) -> dict[str, Any]:
        self.list_requests.append(dict(request))
        prefix = request.get("Prefix", "")
        delimiter = request.get("Delimiter")
        matching_keys = sorted(key for key in self.objects if key.startswith(prefix))
        if delimiter:
            common_prefixes = sorted(
                {
                    f"{prefix}{rest.split(delimiter, 1)[0]}{delimiter}"
                    for key in matching_keys
                    for rest in [key[len(prefix) :]]
                    if delimiter in rest
                }
            )
            return {"CommonPrefixes": [{"Prefix": item} for item in common_prefixes]}
        return {"Contents": [{"Key": key} for key in matching_keys]}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise FakeS3NotFound()
        return {"Body": FakeBody(self.objects[Key])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise FakeS3NotFound()
        return {}


def _manifest(
    *,
    artifact_type: str,
    artifact_id: str,
    metadata: dict[str, Any],
) -> bytes:
    return json.dumps(
        {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "created_at": "2026-05-30T00:00:00Z",
            "created_by": "test",
            "code_git_sha": "abc123",
            "dependencies": [],
            "metadata": metadata,
            "files": [],
        }
    ).encode("utf-8")


def test_view_s3_artifacts_filters_fingerprinted_artifacts(
    monkeypatch: Any,
) -> None:
    client = FakeS3Client(
        {
            "assets/embeddings/new/_MANIFEST.json": _manifest(
                artifact_type="embeddings",
                artifact_id="new",
                metadata={
                    "stage": "embeddings",
                    "asset_fingerprint_sha256": "fp-new",
                    "source_chunked_corpus_artifact_id": "chunks-new",
                },
            ),
            "assets/embeddings/new/_SUCCESS": b"",
            "assets/embeddings/old/_MANIFEST.json": _manifest(
                artifact_type="embeddings",
                artifact_id="old",
                metadata={"stage": "embeddings"},
            ),
            "assets/embeddings/old/_SUCCESS": b"",
            "assets/chunked_corpus/chunks-new/_MANIFEST.json": _manifest(
                artifact_type="chunked_corpus",
                artifact_id="chunks-new",
                metadata={"stage": "chunked_corpus"},
            ),
            "assets/chunked_corpus/chunks-new/_SUCCESS": b"",
        }
    )
    config = SimpleNamespace(s3=SimpleNamespace(bucket="bucket", prefix="assets"))
    captured: dict[str, Any] = {}

    monkeypatch.setattr(viewer_script, "load_config_and_client", lambda path: (config, client))
    monkeypatch.setattr(viewer_script, "redacted_config_summary", lambda config: {"safe": True})
    monkeypatch.setattr(
        viewer_script,
        "output_payload",
        lambda payload, output: captured.update(payload=payload, output=output),
    )

    payload = viewer_script.run(
        argparse.Namespace(
            config=Path("config.yaml"),
            s3_prefix=None,
            artifact_type=["embeddings"],
            artifact_id_contains=None,
            fingerprint="with",
            limit=10,
            include_manifest=False,
            output=None,
        )
    )

    assert payload["artifact_prefix"] == "assets"
    assert payload["stats"]["total"] == 1
    assert payload["stats"]["with_asset_fingerprint"] == 1
    assert payload["artifacts"][0]["artifact_id"] == "new"
    assert payload["artifacts"][0]["complete"] is True
    assert payload["artifacts"][0]["metadata_summary"] == {
        "stage": "embeddings",
        "source_chunked_corpus_artifact_id": "chunks-new",
    }
    assert captured["payload"] == payload
    assert all(request["Delimiter"] == "/" for request in client.list_requests)


def test_view_s3_artifacts_can_filter_missing_fingerprints(
    monkeypatch: Any,
) -> None:
    client = FakeS3Client(
        {
            "assets/embeddings/new/_MANIFEST.json": _manifest(
                artifact_type="embeddings",
                artifact_id="new",
                metadata={"stage": "embeddings", "asset_fingerprint_sha256": "fp-new"},
            ),
            "assets/embeddings/new/_SUCCESS": b"",
            "assets/embeddings/old/_MANIFEST.json": _manifest(
                artifact_type="embeddings",
                artifact_id="old",
                metadata={"stage": "embeddings"},
            ),
            "assets/embeddings/old/_SUCCESS": b"",
        }
    )
    config = SimpleNamespace(s3=SimpleNamespace(bucket="bucket", prefix="assets"))
    monkeypatch.setattr(viewer_script, "load_config_and_client", lambda path: (config, client))
    monkeypatch.setattr(viewer_script, "redacted_config_summary", lambda config: {"safe": True})
    monkeypatch.setattr(viewer_script, "output_payload", lambda payload, output: None)

    payload = viewer_script.run(
        argparse.Namespace(
            config=Path("config.yaml"),
            s3_prefix=None,
            artifact_type=["embeddings"],
            artifact_id_contains=["old"],
            fingerprint="without",
            limit=10,
            include_manifest=True,
            output=None,
        )
    )

    assert payload["stats"]["without_asset_fingerprint"] == 1
    assert payload["artifacts"][0]["artifact_id"] == "old"
    assert payload["artifacts"][0]["manifest"]["artifact_id"] == "old"
