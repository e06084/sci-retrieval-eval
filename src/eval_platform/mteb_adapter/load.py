"""Load MTEB retrieval tasks and export normalized dataset artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.datasets.normalized import write_normalized_dataset_artifact
from eval_platform.datasets.schema import NormalizedDataset
from eval_platform.mteb_adapter.base import (
    MTEBAdapterError,
    extract_generic_retrieval_task_data,
)
from eval_platform.mteb_adapter.registry import NORMALIZER_REGISTRY, get_mteb_task_normalizer


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


def extract_retrieval_data_from_mteb_task(
    task: Any,
    split: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
    """Extract corpus, queries, and qrels from a loaded MTEB task."""
    metadata = getattr(task, "metadata", None)
    task_name = getattr(metadata, "name", None)
    if isinstance(task_name, str) and task_name in NORMALIZER_REGISTRY:
        return NORMALIZER_REGISTRY[task_name].extract_raw(task, split)
    return extract_generic_retrieval_task_data(task, split)


def load_mteb_retrieval_dataset(task_name: str, split: str = "test") -> NormalizedDataset:
    """Load an MTEB retrieval task and normalize it through the registered normalizer."""
    task = load_mteb_task(task_name)
    normalizer = get_mteb_task_normalizer(task_name)
    return normalizer.normalize(task, split=split)


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
