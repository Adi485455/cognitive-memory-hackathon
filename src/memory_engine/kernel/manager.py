"""Controlled mutation boundary for canonical memories."""

from dataclasses import replace
from threading import RLock
from typing import Any

from .events import MemoryDomainEvent, MemoryEventType
from .ids import EventId, MemoryId
from .lifecycle import transition
from .models import Memory, MemoryStatus, utc_now
from .versioning import next_version, validate_expected_version


class MemoryNotFoundError(KeyError):
    """Raised when a requested memory does not exist."""


class MemoryManager:
    """Own canonical memory mutations and emit domain events."""

    def __init__(self) -> None:
        self._memories: dict[MemoryId, Memory] = {}
        self._events: list[MemoryDomainEvent] = []
        self._lock = RLock()

    def create(self, memory: Memory) -> Memory:
        """Create a canonical memory."""
        with self._lock:
            if memory.memory_id in self._memories:
                raise ValueError(
                    f"Memory already exists: {memory.memory_id}"
                )

            self._memories[memory.memory_id] = memory
            self._emit(MemoryEventType.MEMORY_CREATED, memory)
            return memory

    def get(self, memory_id: MemoryId) -> Memory:
        """Retrieve a canonical memory by its ID."""
        with self._lock:
            try:
                return self._memories[memory_id]
            except KeyError as exc:
                raise MemoryNotFoundError(str(memory_id)) from exc

    def list_memories(self) -> tuple[Memory, ...]:
        """Return a snapshot of the current canonical memories."""
        with self._lock:
            return tuple(self._memories.values())

    def list_events(self) -> tuple[MemoryDomainEvent, ...]:
        """Return a snapshot of emitted domain events."""
        with self._lock:
            return tuple(self._events)

    def update(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
        content: str | None = None,
        evidence_ids: tuple | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Update a memory only if its expected version is current."""
        with self._lock:
            current = self.get(memory_id)

            validate_expected_version(
                current.version,
                expected_version,
            )

            if current.status != MemoryStatus.ACTIVE:
                raise ValueError(
                    "Only active memories can be updated"
                )

            updated = replace(
                current,
                content=(
                    current.content if content is None else content
                ),
                evidence_ids=(
                    current.evidence_ids
                    if evidence_ids is None
                    else evidence_ids
                ),
                importance=(
                    current.importance
                    if importance is None
                    else importance
                ),
                confidence=(
                    current.confidence
                    if confidence is None
                    else confidence
                ),
                metadata=(
                    current.metadata
                    if metadata is None
                    else metadata
                ),
                version=next_version(current.version),
                updated_at=utc_now(),
            )

            self._memories[memory_id] = updated
            self._emit(MemoryEventType.MEMORY_UPDATED, updated)

            return updated

    def supersede(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
    ) -> Memory:
        """Mark a memory as superseded without deleting its history."""
        with self._lock:
            current = self.get(memory_id)

            validate_expected_version(
                current.version,
                expected_version,
            )

            new_status = transition(
                current.status,
                MemoryStatus.SUPERSEDED,
            )

            updated = replace(
                current,
                status=new_status,
                version=next_version(current.version),
                updated_at=utc_now(),
            )

            self._memories[memory_id] = updated
            self._emit(
                MemoryEventType.MEMORY_SUPERSEDED,
                updated,
            )

            return updated

    def archive(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
    ) -> Memory:
        """Archive a memory while retaining its canonical record."""
        with self._lock:
            current = self.get(memory_id)

            validate_expected_version(
                current.version,
                expected_version,
            )

            new_status = transition(
                current.status,
                MemoryStatus.ARCHIVED,
            )

            updated = replace(
                current,
                status=new_status,
                version=next_version(current.version),
                updated_at=utc_now(),
            )

            self._memories[memory_id] = updated
            self._emit(
                MemoryEventType.MEMORY_ARCHIVED,
                updated,
            )

            return updated

    def _emit(
        self,
        event_type: MemoryEventType,
        memory: Memory,
    ) -> None:
        """Record an event for each successful canonical mutation."""
        self._events.append(
            MemoryDomainEvent(
                event_id=EventId.new(),
                event_type=event_type,
                memory_id=memory.memory_id,
                memory_version=memory.version,
            )
        )