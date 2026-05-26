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
    VECTOR_DTYPE,
    VECTOR_ENCODING,
    EmbeddedCorpus,
    EmbeddingArtifactError,
    EmbeddingProvenance,
    EmbeddingRecord,
    EmbeddingShard,
    iter_embedding_shards,
    read_embeddings_artifact,
    write_embedding_shards_artifact,
    write_embeddings_artifact,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


class CountingLocalArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.get_file_calls: list[str] = []

    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        self.get_file_calls.append(relative_path)
        return super().get_file(artifact_type, artifact_id, relative_path)


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
    text = payload.decode("utf-8")
    assert text
    assert "vector_b64" in text
    assert '"vector":' not in text
    assert any(file.path == EMBEDDINGS_FILENAME for file in manifest.files)
    assert manifest.files[0].sha256 is not None


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
    assert len(loaded.embeddings) == len(corpus.embeddings)
    for loaded_record, expected_record in zip(loaded.embeddings, corpus.embeddings, strict=True):
        assert loaded_record.chunk_id == expected_record.chunk_id
        assert loaded_record.doc_id == expected_record.doc_id
        assert loaded_record.vector == pytest.approx(expected_record.vector)
        assert loaded_record.metadata == expected_record.metadata
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
    assert manifest.metadata["embedding_dtype"] == VECTOR_DTYPE
    assert manifest.metadata["vector_encoding"] == VECTOR_ENCODING
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
            "embedding_dtype": "wrong",
            "vector_encoding": "wrong",
            "provenance": {"model_name": "wrong"},
            "stage": "embed",
        },
    )
    assert manifest.metadata["embedding_count"] == 3
    assert manifest.metadata["unique_chunk_count"] == 3
    assert manifest.metadata["unique_doc_count"] == 2
    assert manifest.metadata["embedding_dim"] == 2
    assert manifest.metadata["embedding_dtype"] == VECTOR_DTYPE
    assert manifest.metadata["vector_encoding"] == VECTOR_ENCODING
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
    payload = (
        b'{"chunk_id":"chunk-1","doc_id":"doc-1","vector_b64":"zczMPc3MTD4=",'
        b'"vector_encoding":"float32_le_base64","metadata":{}}\n'
    )
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
                "embedding_dtype": VECTOR_DTYPE,
                "vector_encoding": VECTOR_ENCODING,
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
            "embedding_dtype": "wrong",
            "vector_encoding": "wrong",
            "provenance": {"model_name": "wrong"},
            "stage": "embed",
        },
    )
    loaded = read_embeddings_artifact(store, "litsearch_embeddings")
    assert loaded.metadata == {"source": "unit-test", "stage": "embed"}


def test_write_embeddings_artifact_supports_sharded_layout(store: LocalArtifactStore) -> None:
    embeddings = _sample_embedded_corpus()
    manifest = write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        embeddings,
        provenance=_sample_provenance(),
        source_artifact_id="litsearch_chunks",
        shards=[
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                embeddings=embeddings.embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                embeddings=embeddings.embeddings[2:],
            ),
        ],
    )

    assert manifest.metadata["sharding"]["enabled"] is True
    assert len(manifest.metadata["shards"]) == 2
    assert manifest.metadata["source_chunked_corpus_artifact_id"] == "litsearch_chunks"
    assert store.exists("embeddings", "litsearch_embeddings", "embeddings/part-00000.jsonl")
    assert store.exists("embeddings", "litsearch_embeddings", "embeddings/part-00001.jsonl")
    assert not store.exists("embeddings", "litsearch_embeddings", EMBEDDINGS_FILENAME)


def test_iter_embedding_shards_reads_in_manifest_order(store: LocalArtifactStore) -> None:
    embeddings = _sample_embedded_corpus()
    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        embeddings,
        provenance=_sample_provenance(),
        source_artifact_id="litsearch_chunks",
        shards=[
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                embeddings=embeddings.embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                embeddings=embeddings.embeddings[2:],
            ),
        ],
    )

    shards = list(iter_embedding_shards(store, "litsearch_embeddings"))

    assert [shard.shard_id for shard in shards] == ["part-00000", "part-00001"]
    assert [record.chunk_id for record in shards[0].embeddings] == ["chunk-1", "chunk-2"]
    assert [record.chunk_id for record in shards[1].embeddings] == ["chunk-3"]


def test_iter_embedding_shards_is_lazy(tmp_path: Path) -> None:
    store = CountingLocalArtifactStore(tmp_path)
    embeddings = _sample_embedded_corpus()
    write_embeddings_artifact(
        store,
        "litsearch_embeddings",
        embeddings,
        provenance=_sample_provenance(),
        source_artifact_id="litsearch_chunks",
        shards=[
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks/part-00000.jsonl",
                embedding_file="embeddings/part-00000.jsonl",
                source_chunk_count=2,
                embedding_count=2,
                embeddings=embeddings.embeddings[:2],
            ),
            EmbeddingShard(
                shard_id="part-00001",
                source_chunk_file="chunks/part-00001.jsonl",
                embedding_file="embeddings/part-00001.jsonl",
                source_chunk_count=1,
                embedding_count=1,
                embeddings=embeddings.embeddings[2:],
            ),
        ],
    )

    shard_iter = iter_embedding_shards(store, "litsearch_embeddings")

    assert not isinstance(shard_iter, list)
    assert store.get_file_calls == []

    first_shard = next(shard_iter)
    assert first_shard.shard_id == "part-00000"
    assert store.get_file_calls == ["embeddings/part-00000.jsonl"]

    second_shard = next(shard_iter)
    assert second_shard.shard_id == "part-00001"
    assert store.get_file_calls == [
        "embeddings/part-00000.jsonl",
        "embeddings/part-00001.jsonl",
    ]


def test_write_embedding_shards_artifact_writes_first_shard_before_second_generation(
    store: LocalArtifactStore,
) -> None:
    provenance = _sample_provenance()

    class RecordingShardIterable:
        def __init__(self) -> None:
            self.index = 0

        def __iter__(self) -> "RecordingShardIterable":
            return self

        def __next__(self) -> EmbeddingShard:
            if self.index == 0:
                self.index += 1
                return EmbeddingShard(
                    shard_id="part-00000",
                    source_chunk_file="chunks/part-00000.jsonl",
                    embedding_file="embeddings/part-00000.jsonl",
                    source_chunk_count=1,
                    embedding_count=1,
                    embeddings=[
                        EmbeddingRecord(
                            chunk_id="chunk-1",
                            doc_id="doc-1",
                            vector=[0.1, 0.2],
                        )
                    ],
                )
            if self.index == 1:
                assert store.exists(
                    "embeddings",
                    "litsearch_embeddings",
                    "embeddings/part-00000.jsonl",
                )
                self.index += 1
                return EmbeddingShard(
                    shard_id="part-00001",
                    source_chunk_file="chunks/part-00001.jsonl",
                    embedding_file="embeddings/part-00001.jsonl",
                    source_chunk_count=1,
                    embedding_count=1,
                    embeddings=[
                        EmbeddingRecord(
                            chunk_id="chunk-2",
                            doc_id="doc-2",
                            vector=[0.3, 0.4],
                        )
                    ],
                )
            raise StopIteration

    manifest = write_embedding_shards_artifact(
        store,
        "litsearch_embeddings",
        RecordingShardIterable(),
        provenance=provenance,
        source_artifact_id="litsearch_chunks",
    )

    assert manifest.metadata["embedding_count"] == 2
    assert store.is_complete("embeddings", "litsearch_embeddings") is True
