"""Load MTEB retrieval tasks and export normalized dataset artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.datasets.normalized import write_normalized_dataset_artifact
from eval_platform.datasets.schema import NormalizedDataset
from eval_platform.mteb_adapter.convert import convert_retrieval_data_to_normalized_dataset


class MTEBAdapterError(Exception):
    """Raised when MTEB task data cannot be loaded or extracted."""


def load_mteb_task(task_name: str) -> Any:
    """Load a single MTEB task by name."""
    try:
        import mteb
    except ImportError as exc:
        raise ImportError(
            "mteb is required to load MTEB tasks. "
            "Install with: pip install 'sci-retrieval-eval[mteb]'"
        ) from exc

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise MTEBAdapterError(f"MTEB task not found: {task_name}")
    return tasks[0]


def _load_task_data(task: Any, split: str) -> None:
    load_data = getattr(task, "load_data", None)
    if load_data is None:
        return

    attempts = [
        lambda: load_data(eval_splits=[split]),
        lambda: load_data(splits=[split]),
        lambda: load_data(split=split),
        lambda: load_data(),
    ]

    last_type_error: TypeError | None = None
    for attempt in attempts:
        try:
            attempt()
            return
        except TypeError as exc:
            last_type_error = exc

    if last_type_error is not None:
        raise last_type_error


def _select_split(data: Any, split: str) -> Any:
    if isinstance(data, Mapping) and split in data:
        return data[split]
    return data


def _require_mapping(name: str, data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise MTEBAdapterError(f"MTEB task is missing {name}")
    return data


def extract_retrieval_data_from_mteb_task(
    task: Any,
    split: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
    """Extract corpus, queries, and qrels from a loaded MTEB task."""
    _load_task_data(task, split)

    corpus = _select_split(getattr(task, "corpus", None), split)
    queries = _select_split(getattr(task, "queries", None), split)

    qrels = getattr(task, "relevant_docs", None)
    if qrels is None:
        qrels = getattr(task, "qrels", None)
    qrels = _select_split(qrels, split)

    corpus_mapping = _require_mapping("corpus", corpus)
    queries_mapping = _require_mapping("queries", queries)
    qrels_mapping = _require_mapping("qrels", qrels)

    return corpus_mapping, queries_mapping, qrels_mapping


def load_mteb_retrieval_dataset(task_name: str, split: str = "test") -> NormalizedDataset:
    """Load an MTEB retrieval task and convert it to a normalized dataset."""
    task = load_mteb_task(task_name)
    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split)
    return convert_retrieval_data_to_normalized_dataset(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "source": "mteb",
            "task_name": task_name,
            "split": split,
        },
    )


def build_default_artifact_id(task_name: str, split: str, prefix: str = "mteb") -> str:
    """Build a filesystem-safe artifact id for an exported MTEB dataset."""
    raw = f"{prefix}_{task_name}_{split}".lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", raw)
    return normalized.strip("_")


def export_mteb_retrieval_dataset_artifact(
    store: ArtifactStore,
    task_name: str,
    *,
    split: str = "test",
    artifact_id: str | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Load an MTEB retrieval task and write it as a normalized dataset artifact."""
    dataset = load_mteb_retrieval_dataset(task_name, split=split)
    resolved_artifact_id = artifact_id or build_default_artifact_id(task_name, split)

    manifest_metadata: dict[str, Any] = {}
    if metadata:
        manifest_metadata.update(metadata)
    manifest_metadata.update(
        {
            "source": "mteb",
            "task_name": task_name,
            "split": split,
        }
    )

    return write_normalized_dataset_artifact(
        store,
        resolved_artifact_id,
        dataset,
        created_by=created_by,
        code_git_sha=code_git_sha,
        metadata=manifest_metadata,
    )
