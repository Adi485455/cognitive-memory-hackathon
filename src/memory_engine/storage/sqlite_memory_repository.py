"""Transactional SQLite storage for canonical memories and domain events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock

from memory_engine.kernel.events import (
    MemoryDomainEvent,
    MemoryEventType,
)
from memory_engine.kernel.ids import (
    EntityId,
    EventId,
    EvidenceId,
    MemoryId,
)
from memory_engine.kernel.models import (
    Memory,
    MemoryStatus,
    MemoryType,
)
from memory_engine.storage.canonical_repository import (
    MemoryVersionConflictError,
)


class SQLiteMemoryRepository:
    """Persist canonical memories and events transactionally."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._lock = RLock()

        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_events (
                    event_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    memory_version INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_memory
                ON domain_events(memory_id, occurred_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _serialize_memory(memory: Memory) -> str:
        data = {
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "evidence_ids": [
                str(item) for item in memory.evidence_ids
            ],
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "learned_at": memory.learned_at.isoformat(),
            "valid_from": (
                memory.valid_from.isoformat()
                if memory.valid_from
                else None
            ),
            "valid_until": (
                memory.valid_until.isoformat()
                if memory.valid_until
                else None
            ),
            "status": memory.status.value,
            "version": memory.version,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "scope": memory.scope,
            "entity_ids": [
                str(item) for item in memory.entity_ids
            ],
            "supersedes": [
                str(item) for item in memory.supersedes
            ],
            "metadata": memory.metadata,
        }

        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _deserialize_memory(
        memory_id: str,
        record_json: str,
    ) -> Memory:
        try:
            data = json.loads(record_json)

            return Memory(
                memory_id=MemoryId(value=memory_id),
                content=data["content"],
                memory_type=MemoryType(data["memory_type"]),
                evidence_ids=tuple(
                    EvidenceId(value=value)
                    for value in data["evidence_ids"]
                ),
                created_at=datetime.fromisoformat(
                    data["created_at"]
                ),
                updated_at=datetime.fromisoformat(
                    data["updated_at"]
                ),
                learned_at=datetime.fromisoformat(
                    data["learned_at"]
                ),
                valid_from=(
                    datetime.fromisoformat(data["valid_from"])
                    if data.get("valid_from")
                    else None
                ),
                valid_until=(
                    datetime.fromisoformat(data["valid_until"])
                    if data.get("valid_until")
                    else None
                ),
                status=MemoryStatus(data["status"]),
                version=data["version"],
                importance=data["importance"],
                confidence=data["confidence"],
                scope=data["scope"],
                entity_ids=tuple(
                    EntityId(value=value)
                    for value in data.get("entity_ids", [])
                ),
                supersedes=tuple(
                    MemoryId(value=value)
                    for value in data.get("supersedes", [])
                ),
                metadata=data.get("metadata", {}),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                f"Invalid SQLite memory record: {memory_id}"
            ) from exc

    @staticmethod
    def _serialize_event(event: MemoryDomainEvent) -> str:
        data = {
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "memory_id": str(event.memory_id),
            "memory_version": event.memory_version,
            "occurred_at": event.occurred_at.isoformat(),
            "operation_id": event.operation_id,
            "metadata": event.metadata,
        }

        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _deserialize_event(
        record_json: str,
    ) -> MemoryDomainEvent:
        try:
            data = json.loads(record_json)

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
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                "Invalid SQLite domain event record"
            ) from exc

    @staticmethod
    def _validate_event(
        memory: Memory,
        event: MemoryDomainEvent,
    ) -> None:
        if event.memory_id != memory.memory_id:
            raise ValueError(
                "Event memory ID must match the memory"
            )

        if event.memory_version != memory.version:
            raise ValueError(
                "Event version must match the memory"
            )

    def create(
        self,
        memory: Memory,
        event: MemoryDomainEvent,
    ) -> Memory:
        """Atomically create a memory and its creation event."""
        self._validate_event(memory, event)

        if event.event_type != MemoryEventType.MEMORY_CREATED:
            raise ValueError(
                "A new memory requires a MEMORY_CREATED event"
            )

        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                connection.execute(
                    """
                    INSERT INTO memories(
                        memory_id, version, record_json
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(memory.memory_id),
                        memory.version,
                        self._serialize_memory(memory),
                    ),
                )

                self._insert_event(connection, event)
                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return memory

    def update(
        self,
        memory: Memory,
        event: MemoryDomainEvent,
        *,
        expected_version: int,
    ) -> Memory:
        """Atomically update a memory and append its event."""
        self._validate_event(memory, event)

        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    """
                    SELECT version
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (str(memory.memory_id),),
                ).fetchone()

                if row is None:
                    raise KeyError(
                        f"Memory not found: {memory.memory_id}"
                    )

                # Check the stored version first.
                if row["version"] != expected_version:
                    raise MemoryVersionConflictError(
                        f"Expected version {expected_version}; "
                        f"found {row['version']}"
                    )

                # Only a valid, current update reaches this check.
                if memory.version != expected_version + 1:
                    raise ValueError(
                        "Updated version must be expected_version + 1"
                    )

                if event.event_type == MemoryEventType.MEMORY_CREATED:
                    raise ValueError(
                        "An update cannot use MEMORY_CREATED"
                    )

                connection.execute(
                    """
                    UPDATE memories
                    SET version = ?, record_json = ?
                    WHERE memory_id = ?
                    """,
                    (
                        memory.version,
                        self._serialize_memory(memory),
                        str(memory.memory_id),
                    ),
                )

                self._insert_event(connection, event)
                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return memory



    def transition(
        self,
        memory: Memory,
        event: MemoryDomainEvent,
        *,
        expected_version: int,
    ) -> Memory:
        """Atomically persist a lifecycle transition and its event."""
        self._validate_event(memory, event)

        allowed_events = {
            MemoryEventType.MEMORY_SUPERSEDED,
            MemoryEventType.MEMORY_ARCHIVED,
        }

        if event.event_type not in allowed_events:
            raise ValueError(
                "Transition requires a superseded or archived event"
            )

        with self._lock, self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    """
                    SELECT version, record_json
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (str(memory.memory_id),),
                ).fetchone()

                if row is None:
                    raise KeyError(
                        f"Memory not found: {memory.memory_id}"
                    )

                if row["version"] != expected_version:
                    raise MemoryVersionConflictError(
                        f"Expected version {expected_version}; "
                        f"found {row['version']}"
                    )

                if memory.version != expected_version + 1:
                    raise ValueError(
                        "Updated version must be expected_version + 1"
                    )

                current = self._deserialize_memory(
                    str(memory.memory_id),
                    row["record_json"],
                )

                if memory.created_at != current.created_at:
                    raise ValueError("created_at cannot change")

                expected_status = {
                    MemoryEventType.MEMORY_SUPERSEDED: (
                        MemoryStatus.SUPERSEDED
                    ),
                    MemoryEventType.MEMORY_ARCHIVED: (
                        MemoryStatus.ARCHIVED
                    ),
                }[event.event_type]

                if memory.status != expected_status:
                    raise ValueError(
                        "Memory status does not match transition event"
                    )

                connection.execute(
                    """
                    UPDATE memories
                    SET version = ?, record_json = ?
                    WHERE memory_id = ?
                    """,
                    (
                        memory.version,
                        self._serialize_memory(memory),
                        str(memory.memory_id),
                    ),
                )

                self._insert_event(connection, event)
                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return memory






    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event: MemoryDomainEvent,
    ) -> None:
        connection.execute(
            """
            INSERT INTO domain_events(
                event_id,
                memory_id,
                memory_version,
                occurred_at,
                record_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(event.event_id),
                str(event.memory_id),
                event.memory_version,
                event.occurred_at.isoformat(),
                SQLiteMemoryRepository._serialize_event(event),
            ),
        )

    def get(self, memory_id: MemoryId) -> Memory:
        """Retrieve a canonical memory."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM memories
                WHERE memory_id = ?
                """,
                (str(memory_id),),
            ).fetchone()

        if row is None:
            raise KeyError(f"Memory not found: {memory_id}")

        return self._deserialize_memory(
            str(memory_id),
            row["record_json"],
        )

    def list_memories(self) -> tuple[Memory, ...]:
        """Return all canonical memories."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, record_json
                FROM memories
                ORDER BY memory_id
                """
            ).fetchall()

        return tuple(
            self._deserialize_memory(
                row["memory_id"],
                row["record_json"],
            )
            for row in rows
        )

    def list_events(self) -> tuple[MemoryDomainEvent, ...]:
        """Return all events in insertion order."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json
                FROM domain_events
                ORDER BY rowid
                """
            ).fetchall()

        return tuple(
            self._deserialize_event(row["record_json"])
            for row in rows
        )

    def list_events_for_memory(
        self,
        memory_id: MemoryId,
    ) -> tuple[MemoryDomainEvent, ...]:
        """Return events for one memory in insertion order."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_json
                FROM domain_events
                WHERE memory_id = ?
                ORDER BY rowid
                """,
                (str(memory_id),),
            ).fetchall()

        return tuple(
            self._deserialize_event(row["record_json"])
            for row in rows
        )