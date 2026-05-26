"""Tests for raw snapshot to normalized dataset conversion."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking.progress import ProgressEvent
from eval_platform.datasets import (
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
    RawDatasetFile,
    RawDatasetSnapshot,
    RawNormalizeError,
    RawToNormalizedConfig,
    S3RawFileOpener,
    build_content_fingerprint_sha256,
    normalize_raw_dataset_artifact,
    read_normalized_dataset_artifact,
    write_raw_dataset_artifact,
)


class RecordingBinaryStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_calls: list[int] = []
        self.iterated = False

    def read(self, size: int | None = -1) -> bytes:
        recorded_size = -1 if size is None else size
        self.read_calls.append(recorded_size)
        return super().read(size)

    def __iter__(self) -> RecordingBinaryStream:
        self.iterated = True
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if line == b"":
            raise StopIteration
        return line


class FakeRawFileOpener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.streams: dict[str, RecordingBinaryStream] = {}

    def open(self, uri: str) -> RecordingBinaryStream:
        stream = RecordingBinaryStream(self.payloads[uri])
        self.streams[uri] = stream
        return stream


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _ifir_nfcorpus_snapshot() -> tuple[RawDatasetSnapshot, FakeRawFileOpener]:
    uris_to_payloads = {
        "s3://bucket/raw/ifir_nfcorpus/corpus.jsonl": (
            b'{"_id":"d1","title":"Title 1","text":"Doc 1"}\n'
            b'{"_id":"d2","title":"Title 2","text":"Doc 2"}\n'
        ),
        "s3://bucket/raw/ifir_nfcorpus/queries.jsonl": (
            b'{"_id":"q1","text":"Query 1"}\n'
            b'{"_id":"q2","text":"Query 2"}\n'
        ),
        "s3://bucket/raw/ifir_nfcorpus/instructions.jsonl": (
            b'{"query-id":"q1","instruction":"Instruction 1"}\n'
            b'{"query-id":"q2","instruction":"Instruction 2"}\n'
        ),
        "s3://bucket/raw/ifir_nfcorpus/qrels/test.tsv": (
            b"query-id\tcorpus-id\tscore\n"
            b"q1\td1\t1\n"
            b"q2\td2\t2\n"
        ),
    }
    files = [
        RawDatasetFile(
            path="corpus.jsonl",
            uri="s3://bucket/raw/ifir_nfcorpus/corpus.jsonl",
            size_bytes=len(uris_to_payloads["s3://bucket/raw/ifir_nfcorpus/corpus.jsonl"]),
            sha256=(
                "49c683644ef3c9292f95b6a85ee1a29a920d73d034ca1d4f5671bc193d88055b"
            ),
        ),
        RawDatasetFile(
            path="instructions.jsonl",
            uri="s3://bucket/raw/ifir_nfcorpus/instructions.jsonl",
            size_bytes=len(
                uris_to_payloads["s3://bucket/raw/ifir_nfcorpus/instructions.jsonl"]
            ),
            sha256=(
                "1f713e3f812c5a316f50a7d5fd7319189b62efeff6d1b55be295644db7d4fd38"
            ),
        ),
        RawDatasetFile(
            path="qrels/test.tsv",
            uri="s3://bucket/raw/ifir_nfcorpus/qrels/test.tsv",
            size_bytes=len(uris_to_payloads["s3://bucket/raw/ifir_nfcorpus/qrels/test.tsv"]),
            sha256=(
                "63b97907358dba40bde0e166247993eb0dbdd8e5398af0f25b7afdbfb4e4f858"
            ),
        ),
        RawDatasetFile(
            path="queries.jsonl",
            uri="s3://bucket/raw/ifir_nfcorpus/queries.jsonl",
            size_bytes=len(uris_to_payloads["s3://bucket/raw/ifir_nfcorpus/queries.jsonl"]),
            sha256=(
                "7499a884c5cbac43ff8162c89edcf0b08555f2ecf4e7896cb8b4976808470ee1"
            ),
        ),
    ]
    snapshot = RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri="s3://bucket/raw/ifir_nfcorpus",
        dataset_name="IFIRNFCorpus",
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters={"split": "test"},
    )
    return snapshot, FakeRawFileOpener(uris_to_payloads)


def test_normalize_raw_dataset_artifact_ifir_nfcorpus(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    manifest = normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
            metadata={"note": "smoke"},
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, "normalized_ifir_nfcorpus_001")

    assert store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "normalized_ifir_nfcorpus_001")
    assert len(loaded.corpus) == 2
    assert len(loaded.queries) == 2
    assert len(loaded.qrels) == 2
    assert loaded.queries[0].metadata["instruction"] == "Instruction 1"
    assert loaded.qrels[1].relevance == 2.0
    assert manifest.dependencies[0].artifact_type == RAW_DATASET_ARTIFACT_TYPE
    assert manifest.dependencies[0].artifact_id == "raw_ifir_nfcorpus_001"
    assert manifest.metadata["source"] == "raw_dataset"
    assert manifest.metadata["task_name"] == "IFIRNFCorpus"
    assert manifest.metadata["split"] == "test"
    assert manifest.metadata["normalizer_name"] == "ifir_nfcorpus_raw_jsonl_tsv_v1"
    assert manifest.metadata["raw_dataset_artifact_id"] == "raw_ifir_nfcorpus_001"
    assert manifest.metadata["raw_dataset_fingerprint"] == snapshot.content_fingerprint_sha256
    assert manifest.metadata["raw_source_uri"] == "s3://bucket/raw/ifir_nfcorpus"
    assert manifest.metadata["normalized_schema_version"] == "1"
    assert manifest.metadata["note"] == "smoke"


def test_normalize_raw_dataset_artifact_streams_corpus_jsonl(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
        ),
        opener=opener,
    )

    corpus_stream = opener.streams["s3://bucket/raw/ifir_nfcorpus/corpus.jsonl"]
    assert corpus_stream.iterated is True
    assert -1 not in corpus_stream.read_calls


def test_normalize_raw_dataset_artifact_rejects_unknown_normalizer(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    with pytest.raises(RawNormalizeError, match="Unsupported raw normalizer"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_ifir_nfcorpus_001",
                output_artifact_id="normalized_ifir_nfcorpus_001",
                dataset_name="IFIRNFCorpus",
                normalizer_name="unknown",
            ),
            opener=opener,
        )


def test_s3_raw_file_opener_reads_bytes_from_fake_client() -> None:
    client = FakeS3Client()
    client.put_object(Bucket="raw-bucket", Key="path/data.jsonl", Body=b'{"_id":"1"}\n')

    opener = S3RawFileOpener(client=client)

    with opener.open("s3://raw-bucket/path/data.jsonl") as stream:
        assert stream.read() == b'{"_id":"1"}\n'


def test_s3_raw_file_opener_rejects_invalid_uri() -> None:
    opener = S3RawFileOpener(client=FakeS3Client())

    with pytest.raises(RawNormalizeError, match="Unsupported raw file URI"):
        opener.open("file:///tmp/not-s3.jsonl")


def test_normalize_raw_dataset_artifact_reports_progress(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)
    events: list[ProgressEvent] = []

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
        ),
        opener=opener,
        progress_reporter=events.append,
    )

    assert [event.metadata["kind"] for event in events] == [
        "corpus",
        "queries",
        "instructions",
        "qrels",
    ]
    assert events[-1].current == 4
    assert events[-1].total == 4


def test_normalize_raw_dataset_artifact_reporter_failure_does_not_write_success(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    def failing_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_ifir_nfcorpus_001",
                output_artifact_id="normalized_ifir_nfcorpus_001",
                dataset_name="IFIRNFCorpus",
            ),
            opener=opener,
            progress_reporter=failing_reporter,
        )

    assert (
        store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "normalized_ifir_nfcorpus_001")
        is False
    )
