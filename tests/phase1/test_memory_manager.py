import pytest

from memory_engine.kernel.events import MemoryEventType
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.manager import (
    MemoryManager,
    MemoryNotFoundError,
)
from memory_engine.kernel.models import (
    Memory,
    MemoryStatus,
    MemoryType,
)
from memory_engine.kernel.versioning import VersionConflictError


def make_memory():
    return Memory(
        memory_id=MemoryId.new(),
        content="The Memory Manager owns canonical mutations.",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(EvidenceId.new(),),
    )


def test_create_and_retrieve_memory():
    manager = MemoryManager()
    memory = make_memory()

    manager.create(memory)

    assert manager.get(memory.memory_id) == memory


def test_duplicate_creation_is_rejected():
    manager = MemoryManager()
    memory = make_memory()

    manager.create(memory)

    with pytest.raises(ValueError, match="already exists"):
        manager.create(memory)


def test_update_increments_version_and_preserves_evidence():
    manager = MemoryManager()
    original = manager.create(make_memory())

    updated = manager.update(
        original.memory_id,
        expected_version=1,
        content="The Memory Manager controls canonical state.",
    )

    assert updated.version == 2
    assert updated.evidence_ids == original.evidence_ids
    assert updated.content == (
        "The Memory Manager controls canonical state."
    )


def test_stale_update_is_rejected():
    manager = MemoryManager()
    memory = manager.create(make_memory())

    manager.update(
        memory.memory_id,
        expected_version=1,
        confidence=0.9,
    )

    with pytest.raises(VersionConflictError):
        manager.update(
            memory.memory_id,
            expected_version=1,
            confidence=0.8,
        )


def test_archive_preserves_record_and_emits_event():
    manager = MemoryManager()
    memory = manager.create(make_memory())

    archived = manager.archive(
        memory.memory_id,
        expected_version=1,
    )

    assert archived.status == MemoryStatus.ARCHIVED
    assert manager.get(memory.memory_id) == archived
    assert (
        manager.list_events()[-1].event_type
        == MemoryEventType.MEMORY_ARCHIVED
    )


def test_unknown_memory_raises_error():
    manager = MemoryManager()

    with pytest.raises(MemoryNotFoundError):
        manager.get(MemoryId.new())