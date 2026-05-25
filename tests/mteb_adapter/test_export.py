"""Tests for MTEB dataset artifact export."""

from pathlib import Path

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.datasets import NormalizedDataset, QueryRecord, read_normalized_dataset_artifact
from eval_platform.datasets.schema import CorpusRecord, QrelRecord
from eval_platform.mteb_adapter import export_mteb_retrieval_dataset_artifact


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _fake_dataset() -> NormalizedDataset:
    return NormalizedDataset(
        corpus=[CorpusRecord(doc_id="doc-1", text="corpus text")],
        queries=[QueryRecord(query_id="q-1", text="query text")],
        qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
        metadata={"source": "mteb", "task_name": "SciFact", "split": "test"},
    )


def test_export_writes_complete_artifact(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_retrieval_dataset",
        lambda task_name, split="test": _fake_dataset(),
    )

    manifest = export_mteb_retrieval_dataset_artifact(store, "SciFact", split="test")

    assert store.is_complete("normalized_dataset", manifest.artifact_id) is True
    loaded = read_normalized_dataset_artifact(store, manifest.artifact_id)
    assert loaded.corpus[0].doc_id == "doc-1"


def test_export_manifest_metadata_contains_source_task_and_split(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_retrieval_dataset",
        lambda task_name, split="test": _fake_dataset(),
    )

    manifest = export_mteb_retrieval_dataset_artifact(store, "SciFact", split="test")

    assert manifest.metadata["source"] == "mteb"
    assert manifest.metadata["task_name"] == "SciFact"
    assert manifest.metadata["split"] == "test"


def test_export_builds_default_artifact_id(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_retrieval_dataset",
        lambda task_name, split="test": _fake_dataset(),
    )

    manifest = export_mteb_retrieval_dataset_artifact(store, "LitSearchRetrieval", split="test")

    assert manifest.artifact_id == "mteb_litsearchretrieval_test"


def test_export_uses_provided_artifact_id(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_retrieval_dataset",
        lambda task_name, split="test": _fake_dataset(),
    )

    manifest = export_mteb_retrieval_dataset_artifact(
        store,
        "SciFact",
        split="test",
        artifact_id="custom_scifact_test",
    )

    assert manifest.artifact_id == "custom_scifact_test"


def test_export_manifest_system_metadata_is_not_overridden(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_retrieval_dataset",
        lambda task_name, split="test": _fake_dataset(),
    )

    manifest = export_mteb_retrieval_dataset_artifact(
        store,
        "SciFact",
        split="test",
        metadata={"source": "wrong", "task_name": "wrong", "split": "wrong"},
    )

    assert manifest.metadata["source"] == "mteb"
    assert manifest.metadata["task_name"] == "SciFact"
    assert manifest.metadata["split"] == "test"
