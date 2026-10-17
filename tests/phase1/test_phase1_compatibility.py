from memory_engine.kernel.events import MemoryEventType
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.manager import MemoryManager
from memory_engine.kernel.models import (
    Evidence,
    Memory,
    MemoryType,
)


def test_created_and_updated_events_match_memory_versions():
    evidence = Evidence(
        evidence_id=EvidenceId.new(),
        source="conversation",
        content_hash="sha256:compatibility",
    )

    memory = Memory(
        memory_id=MemoryId.new(),
        content="The Dream Engine must preserve provenance.",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(evidence.evidence_id,),
    )

    manager = MemoryManager()
    manager.create(memory)

    updated = manager.update(
        memory.memory_id,
        expected_version=1,
        confidence=0.95,
    )

    events = manager.list_events()

    assert [event.event_type for event in events] == [
        MemoryEventType.MEMORY_CREATED,
        MemoryEventType.MEMORY_UPDATED,
    ]

    assert [event.memory_version for event in events] == [1, 2]
    assert all(event.memory_id == updated.memory_id for event in events)
    assert updated.evidence_ids == (evidence.evidence_id,)


def test_memory_identity_remains_stable_after_update():
    memory = Memory(
        memory_id=MemoryId.new(),
        content="Initial fact",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(EvidenceId.new(),),
    )

    manager = MemoryManager()
    created = manager.create(memory)

    updated = manager.update(
        created.memory_id,
        expected_version=1,
        content="Updated fact",
    )

    assert updated.memory_id == created.memory_id
    assert updated.version == created.version + 1