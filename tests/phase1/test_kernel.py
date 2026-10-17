import pytest

from memory_engine.kernel.ids import (
    EntityId,
    EvidenceId,
    MemoryId,
)
from memory_engine.kernel.models import (
    Entity,
    Evidence,
    Memory,
    MemoryType,
)


def make_evidence():
    return Evidence(
        evidence_id=EvidenceId.new(),
        source="conversation",
        content_hash="sha256:example",
    )


def make_memory():
    evidence = make_evidence()

    memory = Memory(
        memory_id=MemoryId.new(),
        content="The project uses a canonical Memory Kernel.",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(evidence.evidence_id,),
    )

    return memory, evidence


def test_ids_are_unique():
    assert MemoryId.new() != MemoryId.new()


def test_memory_preserves_evidence_reference():
    memory, evidence = make_memory()

    assert memory.evidence_ids == (evidence.evidence_id,)
    assert memory.version == 1


def test_memory_rejects_empty_content():
    evidence = make_evidence()

    with pytest.raises(ValueError, match="content"):
        Memory(
            memory_id=MemoryId.new(),
            content=" ",
            memory_type=MemoryType.SEMANTIC,
            evidence_ids=(evidence.evidence_id,),
        )


def test_entity_requires_a_name():
    with pytest.raises(ValueError, match="name"):
        Entity(
            entity_id=EntityId.new(),
            name=" ",
            entity_type="project",
        )