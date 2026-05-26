"""Reusable progress reporting helpers for long-running pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class ProgressEvent(BaseModel):
    """A structured progress update emitted by pipeline stages."""

    stage: str
    current: int
    total: int | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ProgressReporter = Callable[[ProgressEvent], None]


def report_progress(
    reporter: ProgressReporter | None,
    *,
    stage: str,
    current: int,
    total: int | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit one progress event if a reporter is configured."""

    if reporter is None:
        return
    reporter(
        ProgressEvent(
            stage=stage,
            current=current,
            total=total,
            message=message,
            metadata=dict(metadata or {}),
        )
    )
