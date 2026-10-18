"""Shared dependencies for the Cognitive Memory API."""

import os
from functools import lru_cache
from pathlib import Path

from memory_engine.storage import (
    EvidenceRegistry,
    ObjectStore,
    PersistentMemoryService,
    SQLiteMemoryRepository,
)
from memory_engine.api.orchestrator import MemoryOrchestrator

@lru_cache(maxsize=1)
def get_memory_service() -> PersistentMemoryService:
    """Create the shared persistent memory service."""
    data_dir = Path(
        os.environ.get("MEMORY_DATA_DIR", ".memory_data")
    ).resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    repository = SQLiteMemoryRepository(
        data_dir / "memories.sqlite3"
    )

    return PersistentMemoryService(repository)


@lru_cache(maxsize=1)
def get_evidence_registry() -> EvidenceRegistry:
    """Create the shared evidence registry."""
    data_dir = Path(
        os.environ.get("MEMORY_DATA_DIR", ".memory_data")
    ).resolve()

    object_store = ObjectStore(data_dir / "objects")

    return EvidenceRegistry(
        data_dir / "evidence",
        object_store,
    )

@lru_cache(maxsize=1)
def get_memory_orchestrator() -> MemoryOrchestrator:
    """Create the shared memory orchestration service."""
    return MemoryOrchestrator(
        memory_service=get_memory_service(),
        evidence_registry=get_evidence_registry(),
    )