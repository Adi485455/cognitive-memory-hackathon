"""Stable, strongly typed identifiers for the Memory Kernel."""

from dataclasses import dataclass
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class StableId:
    """Base value object for domain identifiers."""

    value: str

    def __post_init__(self) -> None:
        try:
            UUID(self.value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("ID must contain a valid UUID") from exc

    @classmethod
    def new(cls) -> "StableId":
        return cls(value=str(uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MemoryId(StableId):
    """Identifies a canonical memory."""


@dataclass(frozen=True, slots=True)
class EntityId(StableId):
    """Identifies a canonical entity."""


@dataclass(frozen=True, slots=True)
class RelationshipId(StableId):
    """Identifies a relationship."""


@dataclass(frozen=True, slots=True)
class EvidenceId(StableId):
    """Identifies a source evidence record."""


@dataclass(frozen=True, slots=True)
class EventId(StableId):
    """Identifies a domain event."""


@dataclass(frozen=True, slots=True)
class DecisionId(StableId):
    """Identifies a memory decision."""