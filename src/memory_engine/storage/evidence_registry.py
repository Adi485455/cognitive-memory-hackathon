"""Persistent metadata registry for canonical evidence records."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from memory_engine.kernel.ids import EvidenceId
from memory_engine.kernel.models import Evidence

from .object_store import ObjectStore


class EvidenceRegistry:
    """Persist evidence metadata and its link to raw stored content."""

    def __init__(
        self,
        root: str | Path,
        object_store: ObjectStore,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self._registry_path = self.root / "evidence.json"
        self._object_store = object_store
        self._lock = RLock()

        if not self._registry_path.exists():
            self._write_records({})

    def _read_records(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._registry_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Evidence registry could not be read"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError("Evidence registry must contain a JSON object")

        return data

    def _write_records(
        self,
        records: dict[str, dict[str, Any]],
    ) -> None:
        fd, temporary_name = tempfile.mkstemp(
            prefix=".evidence-",
            suffix=".json",
            dir=self.root,
        )
        temporary_path = Path(temporary_name)

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

            os.replace(temporary_path, self._registry_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def register(
        self,
        content: bytes,
        *,
        source: str,
        source_uri: str | None = None,
        source_span: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        """Store raw content and register its provenance metadata."""
        if not isinstance(content, bytes):
            raise TypeError("Evidence content must be bytes")

        if not source.strip():
            raise ValueError("Evidence source cannot be empty")

        # Ensure metadata can be persisted without losing its JSON meaning.
        safe_metadata = dict(metadata or {})
        try:
            json.dumps(safe_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Evidence metadata must be JSON serializable"
            ) from exc

        # Store raw bytes first; the registry never points to an
        # object that has not already been written successfully.
        digest = self._object_store.put(content)

        with self._lock:
            records = self._read_records()

            # Content-addressed evidence is deduplicated by hash.
            for evidence_id, record in records.items():
                if record["content_hash"] == digest:
                    return self._deserialize(evidence_id, record)

            evidence = Evidence(
                evidence_id=EvidenceId.new(),
                source=source,
                content_hash=digest,
                observed_at=datetime.now(timezone.utc),
                source_uri=source_uri,
                source_span=source_span,
                metadata=safe_metadata,
            )

            records[str(evidence.evidence_id)] = self._serialize(evidence)
            self._write_records(records)

            return evidence

    def get(self, evidence_id: EvidenceId) -> Evidence:
        """Retrieve registered evidence metadata."""
        with self._lock:
            records = self._read_records()
            key = str(evidence_id)

            if key not in records:
                raise KeyError(f"Evidence not found: {key}")

            return self._deserialize(key, records[key])

    def get_content(self, evidence_id: EvidenceId) -> bytes:
        """Retrieve and verify the original evidence bytes."""
        evidence = self.get(evidence_id)
        content = self._object_store.get(evidence.content_hash)

        return content

    def list_evidence(self) -> tuple[Evidence, ...]:
        """Return a snapshot of registered evidence records."""
        with self._lock:
            records = self._read_records()

            return tuple(
                self._deserialize(evidence_id, record)
                for evidence_id, record in records.items()
            )

    @staticmethod
    def _serialize(evidence: Evidence) -> dict[str, Any]:
        return {
            "source": evidence.source,
            "content_hash": evidence.content_hash,
            "observed_at": evidence.observed_at.isoformat(),
            "source_uri": evidence.source_uri,
            "source_span": evidence.source_span,
            "metadata": evidence.metadata,
        }

    @staticmethod
    def _deserialize(
        evidence_id: str,
        record: dict[str, Any],
    ) -> Evidence:
        try:
            return Evidence(
                evidence_id=EvidenceId(value=evidence_id),
                source=record["source"],
                content_hash=record["content_hash"],
                observed_at=datetime.fromisoformat(record["observed_at"]),
                source_uri=record.get("source_uri"),
                source_span=record.get("source_span"),
                metadata=record.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid evidence record: {evidence_id}"
            ) from exc