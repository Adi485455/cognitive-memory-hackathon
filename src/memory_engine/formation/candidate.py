"""Candidate memories produced during formation."""

from dataclasses import dataclass, field
from typing import Any

from memory_engine.kernel.models import MemoryType

from .ingestion import MemoryEvent


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    """A proposed memory awaiting a JEV decision."""

    content: str
    memory_type: MemoryType
    source_event_id: str
    source: str
    scope: str
    importance: float = 0.5
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Candidate content cannot be empty")

        if not self.source_event_id.strip():
            raise ValueError("Candidate must reference its source event")

        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Importance must be between 0 and 1")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

        if not self.scope.strip():
            raise ValueError("Candidate scope cannot be empty")


class CandidateBuilder:
    """Build an initial candidate from a normalized experience."""

    def build(
        self,
        event: MemoryEvent,
        *,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        confidence: float = 0.7,
    ) -> CandidateMemory:
        """Create a candidate while preserving its source reference."""

        return CandidateMemory(
            content=event.content,
            memory_type=memory_type,
            source_event_id=event.event_id,
            source=event.source,
            scope=event.scope,
            importance=importance,
            confidence=confidence,
            metadata={
                **event.metadata,
                "source_occurred_at": event.occurred_at.isoformat(),
            },
        )