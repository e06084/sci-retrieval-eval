"""Tests for embeddings artifact helpers."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_platform.artifacts import (
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    LocalArtifactStore,
)
from eval_platform.embeddings import (
    EMBEDDINGS_FILENAME,
    EmbeddedCorpus,
    EmbeddingArtifactError,
    EmbeddingProvenance,
    EmbeddingRecord,
    read_embeddings_artifact,
    write_embeddings_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _sample_embedded_corpus() -> EmbeddedCorpus:
    return EmbeddedCorpus(
        embeddings=[
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.3, 0.4]),
            EmbeddingRecord(chunk_id="chunk-3", doc_id="doc-2", vector=[0.5, 0.6]),
        ],
        metadata={"source": "unit-test"},
    )


def _sample_provenance() -> EmbeddingProvenance:
    return EmbeddingProvenance(
        model_name="text-embedding-3-large",
        provider="fake",
        embedding_dim=2,
        normalized=True,
    )


def test_write_embeddings_artifact_writes_embeddings_jsonl(store: LocalArtifactStore) -> None:
    manifest = write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
    )

    payload = store.get_file("embeddings", "litsearch_embeddings", EMBEDDINGS_FILENAME)
    assert payload.decode("utf-8")
    assert any(file.path == EMBEDDINGS_FILENAME for file in manifest.files)


def test_write_embeddings_artifact_marks_complete(store: LocalArtifactStore) -> None:
    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
    )
    assert store.is_complete("embeddings", "litsearch_embeddings") is True


def test_read_embeddings_artifact_round_trip(store: LocalArtifactStore) -> None:
    corpus = _sample_embedded_corpus()
    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        corpus,
        provenance=_sample_provenance(),
        metadata={"stage": "embed"},
    )
    loaded = read_embeddings_artifact(store, "litsearch_embeddings")
    assert loaded.embeddings == corpus.embeddings
    assert loaded.metadata == {"source": "unit-test", "stage": "embed"}


def test_write_embeddings_artifact_records_system_metadata(store: LocalArtifactStore) -> None:
    manifest = write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
    )
    assert manifest.metadata["embedding_count"] == 3
    assert manifest.metadata["unique_chunk_count"] == 3
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 2
    assert manifest.metadata["provenance"]["model_name"] == "text-embedding-3-large"


def test_write_embeddings_artifact_system_metadata_overrides_user_values(
    store: LocalArtifactStore,
) -> None:
    manifest = write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
        metadata={
            "embedding_count": 999,
            "unique_chunk_count": 888,
            "unique_doc_count": 777,
            "embedding_dim": 123,
            "provenance": {"model_name": "wrong"},
            "stage": "embed",
        },
    )
    assert manifest.metadata["embedding_count"] == 3
    assert manifest.metadata["unique_chunk_count"] == 3
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 2
    assert manifest.metadata["provenance"]["model_name"] == "text-embedding-3-large"
    assert manifest.metadata["stage"] == "embed"


def test_write_embeddings_artifact_records_source_dependency(store: LocalArtifactStore) -> None:
    manifest = write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
        source_artifact_id="litsearch_chunks",
    )
    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].artifact_id == "litsearch_chunks"
    assert manifest.dependencies[0].artifact_type == "chunked_corpus"


def test_read_embeddings_artifact_requires_success_marker(store: LocalArtifactStore) -> None:
    artifact_id = "incomplete_embeddings"
    payload = b'{"chunk_id":"chunk-1","doc_id":"doc-1","vector":[0.1,0.2]}\n'
    store.put_file("embeddings", artifact_id, EMBEDDINGS_FILENAME, payload)
    store.write_manifest(
        "embeddings",
        artifact_id,
        ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="embeddings",
            created_at=datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC),
            metadata={
                "embedding_count": 1,
                "unique_chunk_count": 1,
                "unique_doc_count": 1,
                "embedding_dim": 2,
                "provenance": _sample_provenance().model_dump(mode="json"),
            },
            files=[ArtifactFile(path=EMBEDDINGS_FILENAME, size_bytes=len(payload))],
        ),
    )

    with pytest.raises(ArtifactIncompleteError):
        read_embeddings_artifact(store, artifact_id)


def test_write_embeddings_artifact_rejects_inconsistent_vector_dimensions(
    store: LocalArtifactStore,
) -> None:
    corpus = EmbeddedCorpus(
        embeddings=[
            EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2]),
            EmbeddingRecord(chunk_id="chunk-2", doc_id="doc-1", vector=[0.3]),
        ]
    )
    with pytest.raises(EmbeddingArtifactError, match="same dimension"):
        write_embeddings_artifact(
            store,
            "litsearch_embeddings",
            corpus,
            provenance=_sample_provenance(),
        )


def test_write_embeddings_artifact_rejects_dimension_mismatch_with_provenance(
    store: LocalArtifactStore,
) -> None:
    with pytest.raises(EmbeddingArtifactError, match="does not match"):
        write_embeddings_artifact(
            store,
            "litsearch_embeddings",
            _sample_embedded_corpus(),
            provenance=EmbeddingProvenance(model_name="model", embedding_dim=3),
        )


def test_read_embeddings_artifact_strips_system_metadata(store: LocalArtifactStore) -> None:
    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        _sample_embedded_corpus(),
        provenance=_sample_provenance(),
        metadata={
            "embedding_count": 999,
            "unique_chunk_count": 999,
            "unique_doc_count": 999,
            "embedding_dim": 999,
            "provenance": {"model_name": "wrong"},
            "stage": "embed",
        },
    )
    loaded = read_embeddings_artifact(store, "litsearch_embeddings")
    assert loaded.metadata == {"source": "unit-test", "stage": "embed"}
