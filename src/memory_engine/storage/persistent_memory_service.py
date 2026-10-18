"""Coordinate canonical memory mutations with durable storage."""

from __future__ import annotations

from threading import RLock

from memory_engine.kernel.events import (
    MemoryDomainEvent,
    MemoryEventType,
)
from memory_engine.kernel.ids import EventId, MemoryId
from memory_engine.kernel.manager import MemoryManager
from memory_engine.kernel.models import Memory
from memory_engine.storage.canonical_repository import (
    CanonicalMemoryRepository,
    MemoryVersionConflictError,
)
from memory_engine.storage.domain_event_store import DomainEventStore
from memory_engine.storage.sqlite_memory_repository import (
    SQLiteMemoryRepository,
)


class PersistentMemoryService:
    """Expose persistent memory operations and durable event history."""

    def __init__(
        self,
        repository: (
            CanonicalMemoryRepository | SQLiteMemoryRepository
        ),
        manager: MemoryManager | None = None,
        event_store: DomainEventStore | None = None,
    ) -> None:
        self.repository = repository
        self.manager = manager or MemoryManager()
        self._lock = RLock()

        self._sqlite_mode = isinstance(
            repository, SQLiteMemoryRepository
        )

        if self._sqlite_mode:
            self.event_store = None
        else:
            self.event_store = event_store or DomainEventStore(
                repository.root
            )

        # Restore existing memories without creating synthetic events.
        for memory in self.repository.list_memories():
            self.manager._memories[memory.memory_id] = memory

    def list_events(self) -> tuple[MemoryDomainEvent, ...]:
        """Return durable domain events."""
        with self._lock:
            if self._sqlite_mode:
                return self.repository.list_events()

            return self.event_store.list_events()

    def list_events_for_memory(
        self,
        memory_id: MemoryId,
    ) -> tuple[MemoryDomainEvent, ...]:
        """Return durable events for a specific memory."""
        with self._lock:
            if self._sqlite_mode:
                return self.repository.list_events_for_memory(
                    memory_id
                )

            return self.event_store.list_events_for_memory(
                memory_id
            )

    def create(self, memory: Memory) -> Memory:
        """Persist a new memory and its creation event."""
        with self._lock:
            event = self._new_event(
                memory,
                MemoryEventType.MEMORY_CREATED,
            )

            if self._sqlite_mode:
                self.repository.create(memory, event)
            else:
                self.repository.create(memory)
                self.event_store.append(event)

            self.manager._memories[memory.memory_id] = memory
            return memory

    def get(self, memory_id: MemoryId) -> Memory:
        """Retrieve a canonical memory."""
        with self._lock:
            return self.repository.get(memory_id)

    def list_memories(self) -> tuple[Memory, ...]:
        """Return all canonical memories."""
        with self._lock:
            return self.repository.list_memories()

    def update(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
        content: str | None = None,
        evidence_ids: tuple | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        """Update a memory and persist its domain event."""
        with self._lock:
            current = self.repository.get(memory_id)
            self._validate_version(current, expected_version)

            temporary = MemoryManager()
            temporary._memories[memory_id] = current

            updated = temporary.update(
                memory_id,
                expected_version=expected_version,
                content=content,
                evidence_ids=evidence_ids,
                importance=importance,
                confidence=confidence,
                metadata=metadata,
            )

            event = temporary.list_events()[-1]

            if self._sqlite_mode:
                self.repository.update(
                    updated,
                    event,
                    expected_version=expected_version,
                )
            else:
                self.repository.update(
                    updated,
                    expected_version=expected_version,
                )
                self.event_store.append(event)

            self.manager._memories[memory_id] = updated
            return updated

    def supersede(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
    ) -> Memory:
        """Supersede a memory and persist its event."""
        return self._transition(
            memory_id,
            expected_version=expected_version,
            operation="supersede",
        )

    def archive(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
    ) -> Memory:
        """Archive a memory and persist its event."""
        return self._transition(
            memory_id,
            expected_version=expected_version,
            operation="archive",
        )

    def _transition(
        self,
        memory_id: MemoryId,
        *,
        expected_version: int,
        operation: str,
    ) -> Memory:
        with self._lock:
            current = self.repository.get(memory_id)
            self._validate_version(current, expected_version)

            temporary = MemoryManager()
            temporary._memories[memory_id] = current

            if operation == "supersede":
                updated = temporary.supersede(
                    memory_id,
                    expected_version=expected_version,
                )
            elif operation == "archive":
                updated = temporary.archive(
                    memory_id,
                    expected_version=expected_version,
                )
            else:
                raise ValueError(
                    f"Unsupported operation: {operation}"
                )

            event = temporary.list_events()[-1]

            if self._sqlite_mode:
                self.repository.transition(
                    updated,
                    event,
                    expected_version=expected_version,
                )
            else:
                self.repository.update(
                    updated,
                    expected_version=expected_version,
                )
                self.event_store.append(event)

            self.manager._memories[memory_id] = updated
            return updated

    @staticmethod
    def _validate_version(
        current: Memory,
        expected_version: int,
    ) -> None:
        if current.version != expected_version:
            raise MemoryVersionConflictError(
                f"Expected version {expected_version}; "
                f"found {current.version}"
            )

    @staticmethod
    def _new_event(
        memory: Memory,
        event_type: MemoryEventType,
    ) -> MemoryDomainEvent:
        return MemoryDomainEvent(
            event_id=EventId.new(),
            event_type=event_type,
            memory_id=memory.memory_id,
            memory_version=memory.version,
        )