"""Tests for thin Python callable external chunker adapter."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.chunking import ChunkRecord
from eval_platform.chunking.external_adapter import (
    ExternalChunkerAdapterError,
    PythonCallableChunkerConfig,
    PythonCallableExternalChunker,
)
from eval_platform.datasets import CorpusRecord, NormalizedDataset, QueryRecord


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _git_available(), reason="git not available")


@pytest.fixture
def dataset() -> NormalizedDataset:
    return NormalizedDataset(
        corpus=[
            CorpusRecord(doc_id="doc-1", text="first document", title="First"),
            CorpusRecord(doc_id="doc-2", text="second document", title="Second"),
        ],
        queries=[QueryRecord(query_id="q-1", text="query")],
        qrels=[],
    )


def _write_module(repo_path: Path, module_name: str, body: str) -> None:
    (repo_path / f"{module_name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def _make_module_name() -> str:
    return f"fake_chunker_{uuid.uuid4().hex}"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("repo_path", ""),
        ("repo_path", " "),
        ("module", ""),
        ("module", " "),
        ("callable_name", ""),
        ("callable_name", " "),
    ],
)
def test_python_callable_chunker_config_rejects_blank_values(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    payload = {
        "repo_path": str(tmp_path),
        "module": "fake_chunker",
        "callable_name": "chunk_dataset",
        field_name: value,
    }
    with pytest.raises(ValidationError):
        PythonCallableChunkerConfig.model_validate(payload)


def test_python_callable_chunker_config_default_kwargs_independent(tmp_path: Path) -> None:
    first = PythonCallableChunkerConfig(
        repo_path=str(tmp_path),
        module="module_a",
        callable_name="callable_a",
    )
    second = PythonCallableChunkerConfig(
        repo_path=str(tmp_path),
        module="module_b",
        callable_name="callable_b",
    )
    first.callable_kwargs["limit"] = 1
    second.callable_kwargs["limit"] = 2
    assert first.callable_kwargs == {"limit": 1}
    assert second.callable_kwargs == {"limit": 2}


def test_python_callable_external_chunker_accepts_chunkrecord_output(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    module_name = _make_module_name()
    _write_module(
        tmp_path,
        module_name,
        """
        from eval_platform.chunking import ChunkRecord

        def chunk_dataset(dataset, prefix="chunk"):
            for doc in dataset.corpus:
                yield ChunkRecord(
                    chunk_id=f"{prefix}-{doc.doc_id}",
                    doc_id=doc.doc_id,
                    text=doc.text,
                    title=doc.title,
                    chunk_index=0,
                )
        """,
    )
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module=module_name,
            callable_name="chunk_dataset",
            callable_kwargs={"prefix": "record"},
        )
    )
    chunks = list(adapter.chunk_corpus(dataset))
    assert all(isinstance(chunk, ChunkRecord) for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == ["record-doc-1", "record-doc-2"]


def test_python_callable_external_chunker_accepts_dict_output(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    module_name = _make_module_name()
    _write_module(
        tmp_path,
        module_name,
        """
        def chunk_dataset(dataset, prefix="chunk"):
            for doc in dataset.corpus:
                yield {
                    "chunk_id": f"{prefix}-{doc.doc_id}",
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "title": doc.title,
                    "chunk_index": 0,
                }
        """,
    )
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module=module_name,
            callable_name="chunk_dataset",
            callable_kwargs={"prefix": "dict"},
        )
    )
    chunks = list(adapter.chunk_corpus(dataset))
    assert [chunk.chunk_id for chunk in chunks] == ["dict-doc-1", "dict-doc-2"]


def test_python_callable_external_chunker_raises_for_invalid_output_type(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    module_name = _make_module_name()
    _write_module(
        tmp_path,
        module_name,
        """
        def chunk_dataset(dataset):
            return [123]
        """,
    )
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module=module_name,
            callable_name="chunk_dataset",
        )
    )
    with pytest.raises(ExternalChunkerAdapterError, match="Unsupported external chunker row type"):
        list(adapter.chunk_corpus(dataset))


def test_python_callable_external_chunker_raises_for_invalid_dict_shape(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    module_name = _make_module_name()
    _write_module(
        tmp_path,
        module_name,
        """
        def chunk_dataset(dataset):
            return [{"doc_id": "doc-1", "text": "body"}]
        """,
    )
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module=module_name,
            callable_name="chunk_dataset",
        )
    )
    with pytest.raises(
        ExternalChunkerAdapterError,
        match="could not be converted to ChunkRecord",
    ):
        list(adapter.chunk_corpus(dataset))


def test_python_callable_external_chunker_raises_when_callable_missing(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    module_name = _make_module_name()
    _write_module(
        tmp_path,
        module_name,
        """
        def other_callable(dataset):
            return []
        """,
    )
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module=module_name,
            callable_name="chunk_dataset",
        )
    )
    with pytest.raises(ExternalChunkerAdapterError, match="callable not found"):
        list(adapter.chunk_corpus(dataset))


def test_python_callable_external_chunker_raises_when_module_missing(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    adapter = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(tmp_path),
            module="missing_chunker_module",
            callable_name="chunk_dataset",
        )
    )
    with pytest.raises(
        ExternalChunkerAdapterError,
        match="Failed to import external chunker module",
    ):
        list(adapter.chunk_corpus(dataset))


def test_python_callable_external_chunker_isolates_same_module_name_across_repos(
    tmp_path: Path,
    dataset: NormalizedDataset,
) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    module_name = "chunker_module"

    sentinel_before = object()
    sys.modules[module_name] = sentinel_before  # type: ignore[assignment]

    _write_module(
        repo_a,
        module_name,
        """
        def chunk_dataset(dataset):
            yield {
                "chunk_id": "a-doc-1",
                "doc_id": dataset.corpus[0].doc_id,
                "text": dataset.corpus[0].text,
                "title": dataset.corpus[0].title,
                "chunk_index": 0,
            }
        """,
    )
    _write_module(
        repo_b,
        module_name,
        """
        def chunk_dataset(dataset):
            yield {
                "chunk_id": "b-doc-1",
                "doc_id": dataset.corpus[0].doc_id,
                "text": dataset.corpus[0].text,
                "title": dataset.corpus[0].title,
                "chunk_index": 0,
            }
        """,
    )

    adapter_a = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(repo_a),
            module=module_name,
            callable_name="chunk_dataset",
        )
    )
    adapter_b = PythonCallableExternalChunker(
        PythonCallableChunkerConfig(
            repo_path=str(repo_b),
            module=module_name,
            callable_name="chunk_dataset",
        )
    )

    chunks_a = list(adapter_a.chunk_corpus(dataset))
    chunks_b = list(adapter_b.chunk_corpus(dataset))

    assert [chunk.chunk_id for chunk in chunks_a] == ["a-doc-1"]
    assert [chunk.chunk_id for chunk in chunks_b] == ["b-doc-1"]
    assert sys.modules[module_name] is sentinel_before
