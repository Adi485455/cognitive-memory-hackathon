"""Tests for evidence object storage and registry persistence."""

import pytest

from memory_engine.kernel.ids import EvidenceId
from memory_engine.storage import EvidenceRegistry, ObjectStore
from memory_engine.kernel.models import Evidence, Memory, MemoryType, MemoryStatus
from datetime import timedelta

import pytest

from memory_engine.kernel.events import (
    MemoryDomainEvent,
    MemoryEventType,
)
from memory_engine.kernel.ids import EventId, MemoryId
from memory_engine.storage.domain_event_store import DomainEventStore


def make_registry(tmp_path):
    object_store = ObjectStore(tmp_path / "objects")
    registry = EvidenceRegistry(
        tmp_path / "registry",
        object_store,
    )
    return object_store, registry


def test_object_store_round_trip_and_deduplication(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    content = b"original evidence"

    first_hash = store.put(content)
    second_hash = store.put(content)

    assert first_hash == second_hash
    assert store.get(first_hash) == content
    assert store.exists(first_hash)


def test_object_store_rejects_invalid_hash(tmp_path):
    store = ObjectStore(tmp_path / "objects")

    with pytest.raises(ValueError):
        store.get("../unsafe-path")


def test_registry_preserves_evidence_metadata_and_content(tmp_path):
    _, registry = make_registry(tmp_path)
    content = b"Aditya completed the memory ingestion milestone."

    evidence = registry.register(
        content,
        source="test",
        source_uri="test://milestone/1",
        source_span="paragraph-1",
        metadata={"category": "milestone"},
    )

    retrieved = registry.get(evidence.evidence_id)

    assert retrieved.evidence_id == evidence.evidence_id
    assert retrieved.content_hash == evidence.content_hash
    assert retrieved.source == "test"
    assert retrieved.source_uri == "test://milestone/1"
    assert retrieved.source_span == "paragraph-1"
    assert retrieved.metadata == {"category": "milestone"}
    assert registry.get_content(evidence.evidence_id) == content


def test_registry_survives_reconstruction(tmp_path):
    content = b"persistent evidence"

    object_store = ObjectStore(tmp_path / "objects")
    registry = EvidenceRegistry(tmp_path / "registry", object_store)

    evidence = registry.register(
        content,
        source="test",
    )

    # Simulate a new registry instance after process restart.
    reopened_store = ObjectStore(tmp_path / "objects")
    reopened_registry = EvidenceRegistry(
        tmp_path / "registry",
        reopened_store,
    )

    recovered = reopened_registry.get(evidence.evidence_id)

    assert recovered == evidence
    assert reopened_registry.get_content(evidence.evidence_id) == content


def test_registry_rejects_empty_source(tmp_path):
    _, registry = make_registry(tmp_path)

    with pytest.raises(ValueError, match="source"):
        registry.register(b"evidence", source=" ")

from dataclasses import replace

from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.models import Memory, MemoryType
from memory_engine.storage import (
    CanonicalMemoryRepository,
    MemoryVersionConflictError,
    ObjectStore,
    PersistentMemoryService
)


def make_memory():
    return Memory(
        memory_id=MemoryId.new(),
        content="The memory repository preserves canonical records.",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(EvidenceId.new(),),
        metadata={"origin": "phase3-test"},
    )


def test_canonical_repository_create_get_and_list(tmp_path):
    repository = CanonicalMemoryRepository(tmp_path / "memories")
    memory = make_memory()

    repository.create(memory)

    assert repository.get(memory.memory_id) == memory
    assert repository.list_memories() == (memory,)


def test_canonical_repository_survives_reconstruction(tmp_path):
    root = tmp_path / "memories"
    repository = CanonicalMemoryRepository(root)
    memory = make_memory()

    repository.create(memory)

    reopened = CanonicalMemoryRepository(root)

    assert reopened.get(memory.memory_id) == memory


def test_canonical_repository_rejects_duplicate_ids(tmp_path):
    repository = CanonicalMemoryRepository(tmp_path / "memories")
    memory = make_memory()

    repository.create(memory)

    with pytest.raises(ValueError, match="already exists"):
        repository.create(memory)


def test_canonical_repository_requires_expected_version(tmp_path):
    repository = CanonicalMemoryRepository(tmp_path / "memories")
    memory = make_memory()
    repository.create(memory)

    updated = replace(
        memory,
        content="Updated canonical memory.",
        version=memory.version + 1,
    )

    with pytest.raises(MemoryVersionConflictError):
        repository.update(updated, expected_version=memory.version + 1)

    assert repository.get(memory.memory_id) == memory


def test_canonical_repository_accepts_valid_versioned_update(tmp_path):
    repository = CanonicalMemoryRepository(tmp_path / "memories")
    memory = make_memory()
    repository.create(memory)

    updated = replace(
        memory,
        content="Updated canonical memory.",
        version=memory.version + 1,
    )

    repository.update(updated, expected_version=memory.version)

    assert repository.get(memory.memory_id) == updated


    from memory_engine.storage import PersistentMemoryService


def test_persistent_service_recovers_canonical_memories(tmp_path):
    repository = CanonicalMemoryRepository(tmp_path / "memories")
    memory = make_memory()
    repository.create(memory)

    service = PersistentMemoryService(repository)

    assert service.get(memory.memory_id) == memory
    assert service.list_memories() == (memory,)


def test_persistent_service_create_survives_restart(tmp_path):
    root = tmp_path / "memories"
    repository = CanonicalMemoryRepository(root)
    service = PersistentMemoryService(repository)
    memory = make_memory()

    service.create(memory)

    reopened = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    assert reopened.get(memory.memory_id) == memory


def test_persistent_service_update_survives_restart(tmp_path):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )
    memory = make_memory()
    service.create(memory)

    updated = service.update(
        memory.memory_id,
        expected_version=memory.version,
        content="Updated durable canonical memory.",
    )

    reopened = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    assert reopened.get(memory.memory_id) == updated
    assert updated.version == memory.version + 1


def test_persistent_service_rejects_stale_update(tmp_path):
    service = PersistentMemoryService(
        CanonicalMemoryRepository(tmp_path / "memories")
    )
    memory = make_memory()
    service.create(memory)

    service.update(
        memory.memory_id,
        expected_version=memory.version,
        content="First update",
    )

    with pytest.raises(MemoryVersionConflictError):
        service.update(
            memory.memory_id,
            expected_version=memory.version,
            content="Stale update",
        )


        from memory_engine.kernel.models import MemoryStatus


def test_persistent_service_supersede_survives_restart(tmp_path):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    memory = make_memory()
    service.create(memory)

    superseded = service.supersede(
        memory.memory_id,
        expected_version=memory.version,
    )

    reopened = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    recovered = reopened.get(memory.memory_id)

    assert recovered == superseded
    assert recovered.status == MemoryStatus.SUPERSEDED
    assert recovered.version == memory.version + 1


def test_persistent_service_archive_survives_restart(tmp_path):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    memory = make_memory()
    service.create(memory)

    archived = service.archive(
        memory.memory_id,
        expected_version=memory.version,
    )

    reopened = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    recovered = reopened.get(memory.memory_id)

    assert recovered == archived
    assert recovered.status == MemoryStatus.ARCHIVED
    assert recovered.version == memory.version + 1


def test_persistent_service_rejects_archiving_stale_version(tmp_path):
    service = PersistentMemoryService(
        CanonicalMemoryRepository(tmp_path / "memories")
    )

    memory = make_memory()
    service.create(memory)

    service.update(
        memory.memory_id,
        expected_version=memory.version,
        content="Updated before archival",
    )

    with pytest.raises(MemoryVersionConflictError):
        service.archive(
            memory.memory_id,
            expected_version=memory.version,
        )

def test_domain_event_survives_store_restart(tmp_path):
    store = DomainEventStore(tmp_path)

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=MemoryId.new(),
        memory_version=1,
        operation_id="operation-1",
        metadata={"source": "test"},
    )

    store.append(event)

    restarted_store = DomainEventStore(tmp_path)
    recovered = restarted_store.list_events()

    assert len(recovered) == 1
    assert recovered[0] == event


def test_domain_event_store_rejects_duplicate_event_id(tmp_path):
    store = DomainEventStore(tmp_path)

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=MemoryId.new(),
        memory_version=1,
    )

    store.append(event)

    with pytest.raises(ValueError, match="Event already exists"):
        store.append(event)


def test_domain_event_store_filters_by_memory(tmp_path):
    store = DomainEventStore(tmp_path)

    target_memory_id = MemoryId.new()

    target_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=target_memory_id,
        memory_version=1,
    )

    unrelated_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=MemoryId.new(),
        memory_version=1,
    )

    store.append_many((target_event, unrelated_event))

    result = store.list_events_for_memory(target_memory_id)

    assert result == (target_event,)


def test_domain_event_store_preserves_event_order(tmp_path):
    store = DomainEventStore(tmp_path)
    memory_id = MemoryId.new()

    first = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory_id,
        memory_version=1,
    )

    second = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_UPDATED,
        memory_id=memory_id,
        memory_version=2,
        occurred_at=first.occurred_at + timedelta(seconds=1),
    )

    store.append_many((first, second))

    assert store.list_events() == (first, second)
def test_persistent_service_persists_creation_event(tmp_path):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    memory = make_memory()
    service.create(memory)

    events = service.list_events()

    assert len(events) == 1
    assert events[0].event_type.value == "memory.created"
    assert events[0].memory_id == memory.memory_id
    assert events[0].memory_version == memory.version


def test_persistent_service_preserves_event_history_after_restart(
    tmp_path,
):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    memory = make_memory()
    service.create(memory)

    service.update(
        memory.memory_id,
        expected_version=memory.version,
        content="Updated content",
    )

    original_events = service.list_events()

    reopened = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    assert reopened.list_events() == original_events
    assert len(reopened.list_events()) == 2


def test_persistent_service_records_archive_event(tmp_path):
    root = tmp_path / "memories"
    service = PersistentMemoryService(
        CanonicalMemoryRepository(root)
    )

    memory = make_memory()
    service.create(memory)

    service.archive(
        memory.memory_id,
        expected_version=memory.version,
    )

    events = service.list_events_for_memory(memory.memory_id)

    assert [event.event_type.value for event in events] == [
        "memory.created",
        "memory.archived",
    ]


    from memory_engine.kernel.events import (
    MemoryDomainEvent,
    MemoryEventType,
)
from memory_engine.kernel.ids import EventId
from memory_engine.storage.sqlite_memory_repository import (
    SQLiteMemoryRepository,
)

def test_sqlite_repository_atomically_creates_memory_and_event(
    tmp_path,
):
    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )

    repository.create(memory, event)

    assert repository.get(memory.memory_id) == memory
    assert repository.list_events() == (event,)


def test_sqlite_repository_persists_memory_and_event_after_restart(
    tmp_path,
):
    database = tmp_path / "memory.sqlite3"
    repository = SQLiteMemoryRepository(database)
    memory = make_memory()

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )

    repository.create(memory, event)

    reopened = SQLiteMemoryRepository(database)

    assert reopened.get(memory.memory_id) == memory
    assert reopened.list_events() == (event,)


def test_sqlite_repository_rejects_stale_update(tmp_path):
    from dataclasses import replace

    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    created_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )
    repository.create(memory, created_event)

    updated = replace(
        memory,
        content="Updated memory",
        version=memory.version + 1,
    )
    update_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_UPDATED,
        memory_id=memory.memory_id,
        memory_version=updated.version,
    )

    with pytest.raises(MemoryVersionConflictError):
        repository.update(
            updated,
            update_event,
            expected_version=memory.version - 1,
        )

    assert repository.get(memory.memory_id) == memory
    assert repository.list_events() == (created_event,)


def test_sqlite_repository_rolls_back_memory_when_event_insert_fails(
    tmp_path,
):
    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )

    repository.create(memory, event)

    updated = replace(
        memory,
        content="This update should roll back",
        version=memory.version + 1,
    )

    # Reuse the existing event ID to force a database constraint failure.
    duplicate_event = MemoryDomainEvent(
        event_id=event.event_id,
        event_type=MemoryEventType.MEMORY_UPDATED,
        memory_id=memory.memory_id,
        memory_version=updated.version,
    )

    with pytest.raises(Exception):
        repository.update(
            updated,
            duplicate_event,
            expected_version=memory.version,
        )

    assert repository.get(memory.memory_id) == memory
    assert repository.list_events() == (event,)

def test_sqlite_repository_atomically_supersedes_memory(tmp_path):
    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    created_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )
    repository.create(memory, created_event)

    from dataclasses import replace
    from memory_engine.kernel.models import MemoryStatus

    superseded = replace(
        memory,
        status=MemoryStatus.SUPERSEDED,
        version=memory.version + 1,
    )

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_SUPERSEDED,
        memory_id=memory.memory_id,
        memory_version=superseded.version,
    )

    repository.transition(
        superseded,
        event,
        expected_version=memory.version,
    )

    assert repository.get(memory.memory_id) == superseded
    assert repository.list_events() == (created_event, event)


def test_sqlite_repository_atomically_archives_memory(tmp_path):
    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    created_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )
    repository.create(memory, created_event)

    from dataclasses import replace
    from memory_engine.kernel.models import MemoryStatus

    archived = replace(
        memory,
        status=MemoryStatus.ARCHIVED,
        version=memory.version + 1,
    )

    event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_ARCHIVED,
        memory_id=memory.memory_id,
        memory_version=archived.version,
    )

    repository.transition(
        archived,
        event,
        expected_version=memory.version,
    )

    assert repository.get(memory.memory_id) == archived
    assert repository.list_events() == (created_event, event)


def test_sqlite_repository_rolls_back_transition_on_duplicate_event(
    tmp_path,
):
    repository = SQLiteMemoryRepository(
        tmp_path / "memory.sqlite3"
    )
    memory = make_memory()

    created_event = MemoryDomainEvent(
        event_id=EventId.new(),
        event_type=MemoryEventType.MEMORY_CREATED,
        memory_id=memory.memory_id,
        memory_version=memory.version,
    )
    repository.create(memory, created_event)

    from dataclasses import replace
    from memory_engine.kernel.models import MemoryStatus

    archived = replace(
        memory,
        status=MemoryStatus.ARCHIVED,
        version=memory.version + 1,
    )

    duplicate_event = MemoryDomainEvent(
        event_id=created_event.event_id,
        event_type=MemoryEventType.MEMORY_ARCHIVED,
        memory_id=memory.memory_id,
        memory_version=archived.version,
    )

    with pytest.raises(Exception):
        repository.transition(
            archived,
            duplicate_event,
            expected_version=memory.version,
        )

    assert repository.get(memory.memory_id) == memory
    assert repository.list_events() == (created_event,)

def test_sqlite_service_creates_memory_and_event_atomically(tmp_path):
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    repository = SQLiteMemoryRepository(
        tmp_path / "service.sqlite3"
    )
    service = PersistentMemoryService(repository)
    memory = make_memory()

    service.create(memory)

    assert service.get(memory.memory_id) == memory
    assert len(service.list_events()) == 1
    assert (
        service.list_events()[0].event_type
        == MemoryEventType.MEMORY_CREATED
    )


def test_sqlite_service_recovers_memories_and_events_after_restart(
    tmp_path,
):
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    database = tmp_path / "service.sqlite3"

    first_service = PersistentMemoryService(
        SQLiteMemoryRepository(database)
    )
    memory = make_memory()
    first_service.create(memory)

    second_service = PersistentMemoryService(
        SQLiteMemoryRepository(database)
    )

    assert second_service.get(memory.memory_id) == memory
    assert second_service.list_events() == first_service.list_events()


def test_sqlite_service_updates_memory_and_event(tmp_path):
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "service.sqlite3")
    )
    memory = make_memory()
    service.create(memory)

    updated = service.update(
        memory.memory_id,
        expected_version=memory.version,
        content="Updated memory content",
    )

    assert updated.content == "Updated memory content"
    assert updated.version == memory.version + 1
    assert service.get(memory.memory_id) == updated
    assert len(service.list_events_for_memory(memory.memory_id)) == 2
    assert (
        service.list_events_for_memory(memory.memory_id)[-1].event_type
        == MemoryEventType.MEMORY_UPDATED
    )


def test_sqlite_service_supersedes_memory_and_persists_event(
    tmp_path,
):
    from memory_engine.kernel.models import MemoryStatus
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "service.sqlite3")
    )
    memory = make_memory()
    service.create(memory)

    superseded = service.supersede(
        memory.memory_id,
        expected_version=memory.version,
    )

    assert superseded.status == MemoryStatus.SUPERSEDED
    assert service.get(memory.memory_id) == superseded
    assert (
        service.list_events_for_memory(memory.memory_id)[-1].event_type
        == MemoryEventType.MEMORY_SUPERSEDED
    )


def test_sqlite_service_archives_memory_and_persists_event(
    tmp_path,
):
    from memory_engine.kernel.models import MemoryStatus
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "service.sqlite3")
    )
    memory = make_memory()
    service.create(memory)

    archived = service.archive(
        memory.memory_id,
        expected_version=memory.version,
    )

    assert archived.status == MemoryStatus.ARCHIVED
    assert service.get(memory.memory_id) == archived
    assert (
        service.list_events_for_memory(memory.memory_id)[-1].event_type
        == MemoryEventType.MEMORY_ARCHIVED
    )