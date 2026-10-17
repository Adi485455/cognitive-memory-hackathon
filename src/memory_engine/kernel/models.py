"""Canonical domain models for the Cognitive Memory system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .ids import EntityId, EvidenceId, MemoryId, RelationshipId


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    TEMPORAL = "temporal"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Reference to the original information supporting a memory."""

    evidence_id: EvidenceId
    source: str
    content_hash: str
    observed_at: datetime = field(default_factory=utc_now)
    source_uri: str | None = None
    source_span: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Evidence source cannot be empty")

        if not self.content_hash.strip():
            raise ValueError("Evidence content_hash cannot be empty")

        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Entity:
    """A person, project, organization, or other identified entity."""

    entity_id: EntityId
    name: str
    entity_type: str
    scope: str = "user"
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Entity name cannot be empty")

        if not self.entity_type.strip():
            raise ValueError("Entity type cannot be empty")


@dataclass(frozen=True, slots=True)
class Relationship:
    """A typed relationship between two entities."""

    relationship_id: RelationshipId
    source_entity_id: EntityId
    target_entity_id: EntityId
    relation_type: str
    confidence: float = 1.0
    evidence_ids: tuple[EvidenceId, ...] = ()
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        if not self.relation_type.strip():
            raise ValueError("Relationship type cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Relationship confidence must be between 0 and 1")

        if self.valid_from and self.valid_until:
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until cannot precede valid_from")


@dataclass(frozen=True, slots=True)
class Memory:
    """Canonical semantic state of one memory."""

    memory_id: MemoryId
    content: str
    memory_type: MemoryType
    evidence_ids: tuple[EvidenceId, ...]

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    learned_at: datetime = field(default_factory=utc_now)

    valid_from: datetime | None = None
    valid_until: datetime | None = None

    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1

    importance: float = 0.5
    confidence: float = 0.5

    scope: str = "user"
    entity_ids: tuple[EntityId, ...] = ()
    supersedes: tuple[MemoryId, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Memory content cannot be empty")

        if not self.evidence_ids:
            raise ValueError("Memory must reference at least one evidence record")

        if self.version < 1:
            raise ValueError("Memory version must be at least 1")

        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Importance must be between 0 and 1")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

        for name in ("created_at", "updated_at", "learned_at"):
            if getattr(self, name).tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")

        if self.valid_from and self.valid_until:
            if self.valid_until < self.valid_from:
                raise ValueError("valid_until cannot precede valid_from")