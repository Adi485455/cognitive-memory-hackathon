"""Durable JSON repository for canonical memory records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from memory_engine.kernel.ids import (
    EntityId,
    EvidenceId,
    MemoryId,
)
from memory_engine.kernel.models import (
    Memory,
    MemoryStatus,
    MemoryType,
    utc_now,
)


class MemoryVersionConflictError(RuntimeError):
    """Raised when a write uses a stale memory version."""


class CanonicalMemoryRepository:
    """Persist canonical memories independently of derived indexes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "memories.json"
        self._lock = RLock()

        if not self._path.exists():
            self._write({})

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Canonical memory repository could not be read"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError("Memory repository must contain a JSON object")

        return data

    def _write(self, records: dict[str, dict[str, Any]]) -> None:
        fd, name = tempfile.mkstemp(
            prefix=".memories-",
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
    def _serialize(memory: Memory) -> dict[str, Any]:
        return {
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "evidence_ids": [str(item) for item in memory.evidence_ids],
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "learned_at": memory.learned_at.isoformat(),
            "valid_from": (
                memory.valid_from.isoformat() if memory.valid_from else None
            ),
            "valid_until": (
                memory.valid_until.isoformat() if memory.valid_until else None
            ),
            "status": memory.status.value,
            "version": memory.version,
            "importance": memory.importance,
            "confidence": memory.confidence,
            "scope": memory.scope,
            "entity_ids": [str(item) for item in memory.entity_ids],
            "supersedes": [str(item) for item in memory.supersedes],
            "metadata": memory.metadata,
        }

    @staticmethod
    def _deserialize(memory_id: str, data: dict[str, Any]) -> Memory:
        from datetime import datetime

        try:
            return Memory(
                memory_id=MemoryId(value=memory_id),
                content=data["content"],
                memory_type=MemoryType(data["memory_type"]),
                evidence_ids=tuple(
                    EvidenceId(value=value)
                    for value in data["evidence_ids"]
                ),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                learned_at=datetime.fromisoformat(data["learned_at"]),
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
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid canonical memory record: {memory_id}"
            ) from exc

    def create(self, memory: Memory) -> Memory:
        """Persist a new canonical memory."""
        with self._lock:
            records = self._read()
            key = str(memory.memory_id)

            if key in records:
                raise ValueError(f"Memory already exists: {key}")

            records[key] = self._serialize(memory)
            self._write(records)
            return memory

    def get(self, memory_id: MemoryId) -> Memory:
        """Retrieve a canonical memory by ID."""
        with self._lock:
            records = self._read()
            key = str(memory_id)

            if key not in records:
                raise KeyError(f"Memory not found: {key}")

            return self._deserialize(key, records[key])

    def list_memories(self) -> tuple[Memory, ...]:
        """Return a snapshot of all canonical memories."""
        with self._lock:
            records = self._read()
            return tuple(
                self._deserialize(key, value)
                for key, value in records.items()
            )

    def update(
        self,
        memory: Memory,
        *,
        expected_version: int,
    ) -> Memory:
        """Replace a memory only when its stored version matches."""
        with self._lock:
            records = self._read()
            key = str(memory.memory_id)

            if key not in records:
                raise KeyError(f"Memory not found: {key}")

            current = self._deserialize(key, records[key])

            if current.version != expected_version:
                raise MemoryVersionConflictError(
                    f"Expected version {expected_version}; "
                    f"found {current.version}"
                )

            if memory.version != expected_version + 1:
                raise ValueError(
                    "Updated memory version must be expected_version + 1"
                )

            if memory.memory_id != current.memory_id:
                raise ValueError("Memory ID cannot change")

            if memory.created_at != current.created_at:
                raise ValueError("created_at cannot change")

            records[key] = self._serialize(memory)
            self._write(records)
            return memory