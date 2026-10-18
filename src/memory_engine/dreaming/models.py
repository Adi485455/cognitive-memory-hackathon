"""Core models for the Dream Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class DreamJobStatus(str, Enum):
    """Lifecycle states for a dream job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DreamJob:
    """A unit of memory-consolidation work."""

    job_id: str = field(default_factory=lambda: str(uuid4()))
    target_memory_ids: tuple[str, ...] = ()
    trigger_event_ids: tuple[str, ...] = ()
    status: DreamJobStatus = DreamJobStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id cannot be empty")

        if self.attempt < 0:
            raise ValueError("attempt cannot be negative")

        if self.status == DreamJobStatus.RUNNING and self.started_at is None:
            raise ValueError("A running job requires started_at")

        if (
            self.status == DreamJobStatus.COMPLETED
            and self.completed_at is None
        ):
            raise ValueError("A completed job requires completed_at")

        for timestamp in (
            self.created_at,
            self.started_at,
            self.completed_at,
        ):
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError("Dream job timestamps must be timezone-aware")


@dataclass(frozen=True, slots=True)
class DreamResult:
    """Outcome of a dream execution."""

    job_id: str
    examined_memory_ids: tuple[str, ...] = ()
    discovered_relationships: int = 0
    duplicate_candidates: int = 0
    conflict_candidates: int = 0
    mutations_applied: int = 0
    warnings: tuple[str, ...] = ()
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id cannot be empty")

        for name in (
            "discovered_relationships",
            "duplicate_candidates",
            "conflict_candidates",
            "mutations_applied",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

        if self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")

        if self.completed_at.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")

        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")