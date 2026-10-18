"""Durable JSON storage for memory domain events."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from memory_engine.kernel.events import (
    MemoryDomainEvent,
    MemoryEventType,
)
from memory_engine.kernel.ids import EventId, MemoryId


class DomainEventStore:
    """Append and retrieve immutable memory domain events."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self._path = self.root / "domain_events.json"
        self._lock = RLock()

        if not self._path.exists():
            self._write([])

    def _read(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Domain event store could not be read"
            ) from exc

        if not isinstance(data, list):
            raise RuntimeError(
                "Domain event store must contain a JSON array"
            )

        return data

    def _write(self, records: list[dict[str, Any]]) -> None:
        fd, name = tempfile.mkstemp(
            prefix=".domain-events-",
            suffix=".json",
            dir=self.root,
        )
        temporary = Path(name)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    records,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _serialize(event: MemoryDomainEvent) -> dict[str, Any]:
        return {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "memory_id": str(event.memory_id),
            "memory_version": event.memory_version,
            "occurred_at": event.occurred_at.isoformat(),
            "operation_id": event.operation_id,
            "metadata": event.metadata,
        }

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> MemoryDomainEvent:
        try:
            return MemoryDomainEvent(
                event_id=EventId(value=data["event_id"]),
                event_type=MemoryEventType(data["event_type"]),
                memory_id=MemoryId(value=data["memory_id"]),
                memory_version=int(data["memory_version"]),
                occurred_at=datetime.fromisoformat(
                    data["occurred_at"]
                ),
                operation_id=data.get("operation_id"),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Invalid domain event record"
            ) from exc

    def append(self, event: MemoryDomainEvent) -> None:
        """Persist an event, rejecting duplicate event IDs."""
        with self._lock:
            records = self._read()
            event_id = str(event.event_id)

            if any(
                record.get("event_id") == event_id
                for record in records
            ):
                raise ValueError(
                    f"Event already exists: {event_id}"
                )

            records.append(self._serialize(event))
            self._write(records)

    def append_many(
        self,
        events: tuple[MemoryDomainEvent, ...],
    ) -> None:
        """Persist multiple events in one file replacement."""
        with self._lock:
            records = self._read()
            known_ids = {
                record.get("event_id")
                for record in records
            }

            for event in events:
                event_id = str(event.event_id)

                if event_id in known_ids:
                    raise ValueError(
                        f"Event already exists: {event_id}"
                    )

                records.append(self._serialize(event))
                known_ids.add(event_id)

            self._write(records)

    def list_events(self) -> tuple[MemoryDomainEvent, ...]:
        """Return stored events in append order."""
        with self._lock:
            return tuple(
                self._deserialize(record)
                for record in self._read()
            )

    def list_events_for_memory(
        self,
        memory_id: MemoryId,
    ) -> tuple[MemoryDomainEvent, ...]:
        """Return events associated with one memory."""
        return tuple(
            event
            for event in self.list_events()
            if event.memory_id == memory_id
        )