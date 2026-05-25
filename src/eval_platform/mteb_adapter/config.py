"""MTEB dataset adapter configuration."""

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

TARGET_MTEB_RETRIEVAL_TASKS = [
    "LitSearchRetrieval",
    "SciFact",
    "IFIRScifact",
    "IFIRNFCorpus",
    "NFCorpus",
]


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class MTEBDatasetExportConfig(BaseModel):
    """Configuration for exporting MTEB retrieval tasks as artifacts."""

    task_names: list[str]
    split: str = "test"
    artifact_id_prefix: str = "mteb"
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_names")
    @classmethod
    def validate_task_names(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("task_names must not be empty")
        return [_non_empty_string(task_name, "task_name") for task_name in value]

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "split")
