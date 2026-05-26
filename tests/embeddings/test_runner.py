"""Tests for embedding runner orchestration."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import ChunkRecord, write_chunked_corpus_artifact
from eval_platform.chunking.schema import ChunkedCorpus
from eval_platform.embeddings import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EMBEDDINGS_FILENAME,
    EmbeddedCorpus,
    EmbeddingConsistencyCheckResult,
    EmbeddingRunConfig,
    EmbeddingRunError,
    FakeEmbeddingClient,
    read_embeddings_artifact,
    run_embedding,
)


class WrongCountEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts[:-1]]


class WrongDimEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


@pytest.fixture
def source_store(tmp_path: Path) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp_path / "source")
    write_chunked_corpus_artifact(
        store,
        "litsearch_chunks",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="first chunk text",
                    title="First",
                    chunk_index=0,
                    metadata={"section": "abstract"},
                ),
                ChunkRecord(
                    chunk_id="chunk-2",
                    doc_id="doc-2",
                    text="second chunk text",
                    chunk_index=1,
                    start_offset=10,
                    end_offset=20,
                ),
            ],
            metadata={"source": "chunking"},
        ),
    )
    return store


@pytest.fixture
def output_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "output")


def _config() -> EmbeddingRunConfig:
    return EmbeddingRunConfig(
        source_artifact_id="litsearch_chunks",
        output_artifact_id="litsearch_embeddings",
        model_name="fake-embedding-model",
        embedding_dim=3,
        provider="fake-provider",
        api_version="v1",
        normalized=True,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_artifact_id", ""),
        ("source_artifact_id", " "),
        ("output_artifact_id", ""),
        ("output_artifact_id", " "),
        ("model_name", ""),
        ("model_name", " "),
    ],
)
def test_embedding_run_config_rejects_blank_values(field_name: str, value: str) -> None:
    payload = {
        "source_artifact_id": "litsearch_chunks",
        "output_artifact_id": "litsearch_embeddings",
        "model_name": "fake-model",
        "embedding_dim": 3,
        field_name: value,
    }
    with pytest.raises(ValidationError):
        EmbeddingRunConfig.model_validate(payload)


def test_embedding_run_config_default_metadata_not_shared() -> None:
    first = EmbeddingRunConfig(
        source_artifact_id="a",
        output_artifact_id="b",
        model_name="model-a",
        embedding_dim=3,
    )
    second = EmbeddingRunConfig(
        source_artifact_id="c",
        output_artifact_id="d",
        model_name="model-b",
        embedding_dim=3,
    )
    first.metadata["run"] = 1
    second.metadata["run"] = 2
    assert first.metadata == {"run": 1}
    assert second.metadata == {"run": 2}


def test_fake_embedding_client_is_deterministic() -> None:
    client = FakeEmbeddingClient(embedding_dim=3)
    texts = ["a", "b"]
    assert client.embed_texts(texts) == client.embed_texts(texts)


def test_run_embedding_local_to_local_round_trip(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_embedding(source_store, output_store, _config(), FakeEmbeddingClient(3))
    loaded = read_embeddings_artifact(output_store, "litsearch_embeddings")

    assert manifest.artifact_id == "litsearch_embeddings"
    assert output_store.is_complete("embeddings", "litsearch_embeddings") is True
    assert isinstance(loaded, EmbeddedCorpus)
    assert len(loaded.embeddings) == 2
    assert loaded.embeddings[0].chunk_id == "chunk-1"
    assert loaded.embeddings[0].doc_id == "doc-1"
    assert loaded.embeddings[0].metadata["section"] == "abstract"
    assert loaded.embeddings[0].metadata["title"] == "First"
    assert loaded.embeddings[1].metadata["start_offset"] == 10
    assert loaded.embeddings[1].metadata["end_offset"] == 20


def test_run_embedding_records_source_dependency(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    manifest = run_embedding(source_store, output_store, _config(), FakeEmbeddingClient(3))
    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].artifact_id == "litsearch_chunks"
    assert manifest.dependencies[0].artifact_type == "chunked_corpus"


def test_run_embedding_records_provenance_metadata(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_id = "endpoint-a"
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]
    config.batch_size = 8
    config.timeout_seconds = 60.0
    config.consistency_check = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        passed=True,
        max_abs_diff=0.0,
    )
    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    provenance = manifest.metadata["provenance"]
    assert provenance["model_name"] == "fake-embedding-model"
    assert provenance["provider"] == "fake-provider"
    assert provenance["api_version"] == "v1"
    assert provenance["embedding_dim"] == 3
    assert provenance["normalized"] is True
    assert provenance["endpoint_id"] == "endpoint-a"
    assert provenance["endpoint_ids"] == ["endpoint-a", "endpoint-b"]
    assert provenance["consistency_check"]["passed"] is True
    assert provenance["runtime_parameters"]["batch_size"] == 8
    assert provenance["runtime_parameters"]["timeout_seconds"] == 60.0
    assert manifest.metadata["embedding_count"] == 2
    assert manifest.metadata["unique_chunk_count"] == 2
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 3


def test_run_embedding_runner_metadata_cannot_override_system_fields(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.metadata = {
        "embedding_count": 999,
        "unique_chunk_count": 999,
        "unique_doc_count": 999,
        "embedding_dim": 999,
        "provenance": {"model_name": "wrong"},
        "stage": "embedding",
    }

    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert manifest.metadata["embedding_count"] == 2
    assert manifest.metadata["unique_chunk_count"] == 2
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 3
    assert manifest.metadata["stage"] == "embedding"
    assert manifest.metadata["provenance"]["model_name"] == "fake-embedding-model"


def test_run_embedding_raises_for_wrong_vector_count(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    with pytest.raises(EmbeddingRunError, match="different number of vectors"):
        run_embedding(source_store, output_store, _config(), WrongCountEmbeddingClient())
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_for_wrong_vector_dimension(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    with pytest.raises(EmbeddingRunError, match="unexpected dimension"):
        run_embedding(source_store, output_store, _config(), WrongDimEmbeddingClient())
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_when_consistency_check_failed(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]
    config.consistency_check = EmbeddingConsistencyCheckResult(
        input_text="probe text",
        endpoint_ids=["endpoint-a", "endpoint-b"],
        passed=False,
        failure_reason="endpoint mismatch",
        max_abs_diff=0.1,
    )

    with pytest.raises(EmbeddingRunError, match="consistency check failed"):
        run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_raises_when_multi_endpoint_check_missing(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_ids = ["endpoint-a", "endpoint-b"]

    with pytest.raises(EmbeddingRunError, match="require a consistency_check result"):
        run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is False
    assert not output_store.exists(
        EMBEDDINGS_ARTIFACT_TYPE,
        "litsearch_embeddings",
        EMBEDDINGS_FILENAME,
    )


def test_run_embedding_allows_single_endpoint_without_consistency_check(
    source_store: LocalArtifactStore,
    output_store: LocalArtifactStore,
) -> None:
    config = _config()
    config.endpoint_id = "endpoint-a"
    config.endpoint_ids = ["endpoint-a"]

    manifest = run_embedding(source_store, output_store, config, FakeEmbeddingClient(3))

    assert manifest.artifact_id == "litsearch_embeddings"
    assert output_store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, "litsearch_embeddings") is True
