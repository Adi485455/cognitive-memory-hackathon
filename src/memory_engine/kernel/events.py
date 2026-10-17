"""Domain events describing canonical memory changes."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import EventId, MemoryId
from .models import utc_now


class MemoryEventType(str, Enum):
    MEMORY_CREATED = "memory.created"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_SUPERSEDED = "memory.superseded"
    MEMORY_ARCHIVED = "memory.archived"
    MEMORY_REINFORCED = "memory.reinforced"
    MEMORY_RELATIONSHIP_CREATED = "memory.relationship_created"


@dataclass(frozen=True, slots=True)
class MemoryDomainEvent:
    """An immutable record of a canonical memory operation."""

    event_id: EventId
    event_type: MemoryEventType
    memory_id: MemoryId
    memory_version: int

    occurred_at: datetime = field(default_factory=utc_now)
    operation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.memory_version < 1:
            raise ValueError("Event memory_version must be at least 1")

        if self.occurred_at.tzinfo is None:
            raise ValueError("Event timestamp must be timezone-aware")