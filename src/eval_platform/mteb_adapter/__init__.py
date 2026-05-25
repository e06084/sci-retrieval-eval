"""MTEB integration layer."""

from eval_platform.mteb_adapter.config import TARGET_MTEB_RETRIEVAL_TASKS, MTEBDatasetExportConfig
from eval_platform.mteb_adapter.convert import (
    MTEBConversionError,
    convert_retrieval_data_to_normalized_dataset,
)
from eval_platform.mteb_adapter.load import (
    MTEBAdapterError,
    build_default_artifact_id,
    export_mteb_retrieval_dataset_artifact,
    extract_retrieval_data_from_mteb_task,
    load_mteb_retrieval_dataset,
    load_mteb_task,
)

__all__ = [
    "MTEBAdapterError",
    "MTEBConversionError",
    "MTEBDatasetExportConfig",
    "TARGET_MTEB_RETRIEVAL_TASKS",
    "build_default_artifact_id",
    "convert_retrieval_data_to_normalized_dataset",
    "export_mteb_retrieval_dataset_artifact",
    "extract_retrieval_data_from_mteb_task",
    "load_mteb_retrieval_dataset",
    "load_mteb_task",
]
