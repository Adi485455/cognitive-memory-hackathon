"""Tests for memory orchestration and JEV decision handling."""

import pytest

from memory_engine.api.orchestrator import MemoryOrchestrator
from memory_engine.kernel.models import MemoryType
from memory_engine.storage import (
    EvidenceRegistry,
    ObjectStore,
    PersistentMemoryService,
    SQLiteMemoryRepository,
)


@pytest.fixture
def orchestrator(tmp_path):
    """Create an orchestrator backed by isolated persistent storage."""

    data_dir = tmp_path / "memory_data"
    data_dir.mkdir()

    repository = SQLiteMemoryRepository(
        data_dir / "memories.sqlite3"
    )
    memory_service = PersistentMemoryService(repository)

    evidence_registry = EvidenceRegistry(
        data_dir / "evidence",
        ObjectStore(data_dir / "objects"),
    )

    return MemoryOrchestrator(
        memory_service=memory_service,
        evidence_registry=evidence_registry,
    )


def test_valid_experience_creates_canonical_memory(orchestrator):
    result = orchestrator.process(
        "The user prefers C++ for DSA practice.",
        source="phase5-test",
        scope="test-user",
        memory_type=MemoryType.PREFERENCE,
        importance=0.8,
        confidence=0.9,
    )

    assert result.action == "store"
    assert result.requires_review is False
    assert result.memory is not None
    assert result.evidence_id is not None

    assert result.memory.content == (
        "The user prefers C++ for DSA practice."
    )
    assert result.memory.memory_type == MemoryType.PREFERENCE
    assert result.memory.scope == "test-user"
    assert len(result.memory.evidence_ids) == 1

    assert str(result.memory.evidence_ids[0]) == result.evidence_id


def test_successful_storage_records_creation_event(orchestrator):
    result = orchestrator.process(
        "The user prefers detailed technical explanations.",
        source="phase5-test",
        scope="test-user",
        confidence=0.9,
    )

    assert result.memory is not None

    events = orchestrator.memory_service.list_events_for_memory(
        result.memory.memory_id
    )

    assert len(events) == 1
    assert events[0].memory_version == 1


def test_duplicate_experience_is_not_stored_twice(orchestrator):
    content = "The user prefers C++ for DSA practice."

    first = orchestrator.process(
        content,
        source="phase5-test",
        scope="test-user",
        confidence=0.9,
    )

    second = orchestrator.process(
        content,
        source="phase5-test",
        scope="test-user",
        confidence=0.9,
    )

    assert first.action == "store"
    assert first.memory is not None

    assert second.action == "ignore"
    assert second.memory is None
    assert second.requires_review is False

    memories = orchestrator.memory_service.list_memories()

    assert len(memories) == 1


def test_duplicate_detection_ignores_case_and_extra_whitespace(
    orchestrator,
):
    orchestrator.process(
        "The user prefers C++ for DSA practice.",
        scope="test-user",
        confidence=0.9,
    )

    result = orchestrator.process(
        "  THE USER PREFERS   C++ FOR DSA PRACTICE.  ",
        scope="test-user",
        confidence=0.9,
    )

    assert result.action == "ignore"
    assert result.memory is None

    assert len(orchestrator.memory_service.list_memories()) == 1


def test_low_confidence_experience_requires_review(orchestrator):
    result = orchestrator.process(
        "The user might prefer a different programming language.",
        source="phase5-test",
        scope="test-user",
        confidence=0.2,
    )

    assert result.action == "uncertain"
    assert result.requires_review is True
    assert result.memory is None
    assert result.evidence_id is None

    assert orchestrator.memory_service.list_memories() == ()
    assert orchestrator.memory_service.list_events() == ()


def test_confidence_at_threshold_is_not_rejected_as_low(
    orchestrator,
):
    result = orchestrator.process(
        "The user prefers structured technical explanations.",
        source="phase5-test",
        scope="test-user",
        confidence=0.3,
    )

    assert result.action == "store"
    assert result.requires_review is False
    assert result.memory is not None


def test_same_content_in_different_scopes_is_stored_separately(
    orchestrator,
):
    content = "The user prefers C++ for DSA practice."

    first = orchestrator.process(
        content,
        scope="user-a",
        confidence=0.9,
    )

    second = orchestrator.process(
        content,
        scope="user-b",
        confidence=0.9,
    )

    assert first.action == "store"
    assert second.action == "store"

    assert first.memory is not None
    assert second.memory is not None

    assert first.memory.scope == "user-a"
    assert second.memory.scope == "user-b"

    assert first.memory.memory_id != second.memory.memory_id
    assert len(orchestrator.memory_service.list_memories()) == 2


def test_orchestrator_preserves_source_metadata(orchestrator):
    result = orchestrator.process(
        "The user prefers concise code examples.",
        source="phase5-test",
        scope="test-user",
        confidence=0.9,
        metadata={"test_case": "source-provenance"},
    )

    assert result.memory is not None

    memory = result.memory

    assert memory.metadata["source"] == "phase5-test"
    assert memory.metadata["test_case"] == "source-provenance"
    assert memory.metadata["source_event_id"] == result.event_id
    assert memory.metadata["evidence_content_hash"]