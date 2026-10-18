"""Discover potential duplicates, relationships, and conflicts."""

from __future__ import annotations

from dataclasses import dataclass

from memory_engine.kernel.models import Memory


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """Two memories that may express the same information."""

    first_memory_id: str
    second_memory_id: str
    similarity: float
    reason: str

    def __post_init__(self) -> None:
        if self.first_memory_id == self.second_memory_id:
            raise ValueError("Duplicate candidates must be different memories")

        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("similarity must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RelationshipCandidate:
    """A possible relationship between two memories."""

    first_memory_id: str
    second_memory_id: str
    relation_type: str
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if self.first_memory_id == self.second_memory_id:
            raise ValueError("Relationship candidates must be different memories")

        if not self.relation_type.strip():
            raise ValueError("relation_type cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ConflictCandidate:
    """Two memories whose claims may contradict each other."""

    first_memory_id: str
    second_memory_id: str
    reason: str
    confidence: float

    def __post_init__(self) -> None:
        if self.first_memory_id == self.second_memory_id:
            raise ValueError("Conflict candidates must be different memories")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Immutable report produced by memory discovery."""

    duplicate_candidates: tuple[DuplicateCandidate, ...] = ()
    relationship_candidates: tuple[RelationshipCandidate, ...] = ()
    conflict_candidates: tuple[ConflictCandidate, ...] = ()
    examined_memory_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()