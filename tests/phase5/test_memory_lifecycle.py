"""Tests for canonical memory lifecycle and version control."""

from uuid import uuid4

import pytest

from memory_engine.kernel.events import MemoryEventType
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.models import Memory, MemoryType
from memory_engine.storage import (
    PersistentMemoryService,
    SQLiteMemoryRepository,
)
from memory_engine.storage.canonical_repository import (
    MemoryVersionConflictError,
)


@pytest.fixture
def memory_service(tmp_path):
    """Create a persistent memory service backed by isolated SQLite storage."""
    repository = SQLiteMemoryRepository(
        tmp_path / "memories.sqlite3"
    )

    return PersistentMemoryService(repository)


@pytest.fixture
def sample_memory():
    """Create a canonical memory with evidence provenance."""
    return Memory(
        memory_id=MemoryId.new(),
        content="The user prefers C++ for DSA practice.",
        memory_type=MemoryType.PREFERENCE,
        evidence_ids=(EvidenceId.new(),),
        importance=0.8,
        confidence=0.9,
        scope="test",
        metadata={"source": "phase5-test"},
    )


def create_sample_memory(memory_service, sample_memory):
    """Persist a sample memory and return the stored instance."""
    return memory_service.create(sample_memory)


def test_create_memory_starts_at_initial_version(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    assert created.memory_id == sample_memory.memory_id
    assert created.version == 1
    assert created.content == sample_memory.content

    stored = memory_service.get(created.memory_id)

    assert stored.memory_id == created.memory_id
    assert stored.version == 1


def test_update_increments_memory_version(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    updated = memory_service.update(
        created.memory_id,
        expected_version=1,
        content="The user prefers C++ for advanced DSA practice.",
    )

    assert updated.version == 2
    assert updated.content == (
        "The user prefers C++ for advanced DSA practice."
    )

    stored = memory_service.get(created.memory_id)

    assert stored.version == 2
    assert stored.content == updated.content


def test_update_preserves_evidence_ids(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    updated = memory_service.update(
        created.memory_id,
        expected_version=1,
        content="Updated memory content.",
    )

    assert updated.evidence_ids == created.evidence_ids


def test_stale_version_update_is_rejected(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    memory_service.update(
        created.memory_id,
        expected_version=1,
        content="First update.",
    )

    with pytest.raises(MemoryVersionConflictError):
        memory_service.update(
            created.memory_id,
            expected_version=1,
            content="This stale update must fail.",
        )

    stored = memory_service.get(created.memory_id)

    assert stored.version == 2
    assert stored.content == "First update."


def test_create_and_update_generate_durable_events(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    memory_service.update(
        created.memory_id,
        expected_version=1,
        content="Updated canonical memory.",
    )

    events = memory_service.list_events_for_memory(
        created.memory_id
    )

    assert len(events) == 2

    assert events[0].event_type == MemoryEventType.MEMORY_CREATED
    assert events[0].memory_version == 1

    assert events[1].memory_version == 2


def test_archive_transitions_memory_status_and_records_event(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    archived = memory_service.archive(
        created.memory_id,
        expected_version=1,
    )

    assert archived.version == 2

    stored = memory_service.get(created.memory_id)

    assert stored.version == 2
    assert stored.status == archived.status

    events = memory_service.list_events_for_memory(
        created.memory_id
    )

    assert len(events) == 2
    assert events[-1].memory_version == 2


def test_supersede_transitions_memory_and_records_event(
    memory_service,
    sample_memory,
):
    created = create_sample_memory(
        memory_service,
        sample_memory,
    )

    superseded = memory_service.supersede(
        created.memory_id,
        expected_version=1,
    )

    assert superseded.version == 2

    stored = memory_service.get(created.memory_id)

    assert stored.version == 2
    assert stored.status == superseded.status

    events = memory_service.list_events_for_memory(
        created.memory_id
    )

    assert len(events) == 2
    assert events[-1].memory_version == 2


def test_update_missing_memory_raises_key_error(
    memory_service,
):
    missing_id = MemoryId(value=str(uuid4()))

    with pytest.raises(KeyError):
        memory_service.update(
            missing_id,
            expected_version=1,
            content="This memory does not exist.",
        )