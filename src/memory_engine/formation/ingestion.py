"""Normalize incoming experiences for memory formation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """A normalized experience entering the memory system."""

    event_id: str
    content: str
    source: str
    occurred_at: datetime
    scope: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("Event ID cannot be empty")

        if not self.content.strip():
            raise ValueError("Event content cannot be empty")

        if not self.source.strip():
            raise ValueError("Event source cannot be empty")

        if not self.scope.strip():
            raise ValueError("Event scope cannot be empty")

        if self.occurred_at.tzinfo is None:
            raise ValueError("Event timestamp must be timezone-aware")


class ExperienceIngestor:
    """Convert raw experiences into normalized memory events."""

    def ingest(
        self,
        content: str,
        *,
        source: str = "conversation",
        scope: str = "user",
        occurred_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEvent:
        """Validate and normalize one incoming experience."""

        if not isinstance(content, str):
            raise TypeError("Content must be a string")

        normalized_content = " ".join(content.split())

        if not normalized_content:
            raise ValueError("Experience content cannot be empty")

        timestamp = occurred_at or utc_now()

        if timestamp.tzinfo is None:
            raise ValueError("Experience timestamp must be timezone-aware")

        return MemoryEvent(
            event_id=str(uuid4()),
            content=normalized_content,
            source=source.strip(),
            occurred_at=timestamp,
            scope=scope.strip(),
            metadata=dict(metadata or {}),
        )