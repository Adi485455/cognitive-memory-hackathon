from memory_engine.dreaming import (
    DreamConfig,
    DreamEngine,
    DreamJob,
    DreamJobStatus,
    DreamResult,
)
from memory_engine.kernel.manager import MemoryManager
from memory_engine.storage.persistent_memory_service import (
    PersistentMemoryService,
)


def test_dream_engine_imports_without_replacing_memory_manager():
    assert MemoryManager is not None
    assert PersistentMemoryService is not None
    assert DreamEngine is not None


def test_dreaming_public_api_is_available():
    config = DreamConfig()
    engine = DreamEngine(config)
    job = DreamJob()

    assert job.status == DreamJobStatus.PENDING
    assert engine.config is config
    assert DreamResult(job_id=job.job_id).job_id == job.job_id

def test_dream_result_store_persists_payload(tmp_path):
    from memory_engine.dreaming import DreamResultStore

    database_path = tmp_path / "dream_results.sqlite3"
    store = DreamResultStore(database_path)

    payload = {
        "result": {
            "job_id": "job-123",
            "examined_memory_ids": ["memory-1", "memory-2"],
            "mutations_applied": 0,
        },
        "discovery": {
            "duplicate_candidates": [],
            "warnings": ["No conflicts detected"],
        },
    }

    store.save("job-123", payload)

    assert store.exists("job-123")
    assert store.get("job-123") == payload


def test_dream_result_store_survives_reopening_database(tmp_path):
    from memory_engine.dreaming import DreamResultStore

    database_path = tmp_path / "dream_results.sqlite3"
    payload = {
        "job_id": "job-456",
        "mutations_applied": 0,
        "warnings": [],
    }

    first_store = DreamResultStore(database_path)
    first_store.save("job-456", payload)

    reopened_store = DreamResultStore(database_path)

    assert reopened_store.get("job-456") == payload


def test_dream_result_store_rejects_conflicting_duplicate(tmp_path):
    import pytest

    from memory_engine.dreaming import DreamResultStore

    store = DreamResultStore(tmp_path / "dream_results.sqlite3")

    store.save("job-789", {"mutations_applied": 0})

    with pytest.raises(ValueError, match="different result already exists"):
        store.save("job-789", {"mutations_applied": 1})

    assert store.get("job-789") == {"mutations_applied": 0}

def test_dream_execution_serialization_round_trip():
    from datetime import datetime, timezone

    from memory_engine.decision.jev import JEVAction, JEVDecision
    from memory_engine.dreaming.consolidation import ConsolidationResult
    from memory_engine.dreaming.discovery import (
        ConflictCandidate,
        DiscoveryResult,
        DuplicateCandidate,
        RelationshipCandidate,
    )
    from memory_engine.dreaming.models import DreamResult
    from memory_engine.dreaming.serialization import (
        dream_execution_from_dict,
        dream_execution_to_dict,
    )

    started_at = datetime(2026, 10, 18, 10, 0, tzinfo=timezone.utc)
    completed_at = datetime(2026, 10, 18, 10, 1, tzinfo=timezone.utc)

    result = DreamResult(
        job_id="job-round-trip",
        examined_memory_ids=("memory-1", "memory-2"),
        discovered_relationships=1,
        duplicate_candidates=1,
        conflict_candidates=1,
        mutations_applied=0,
        warnings=("Review required",),
        started_at=started_at,
        completed_at=completed_at,
    )

    discovery = DiscoveryResult(
        duplicate_candidates=(
            DuplicateCandidate(
                first_memory_id="memory-1",
                second_memory_id="memory-2",
                similarity=0.97,
                reason="Highly similar content",
            ),
        ),
        relationship_candidates=(
            RelationshipCandidate(
                first_memory_id="memory-1",
                second_memory_id="memory-2",
                relation_type="related_to",
                confidence=0.91,
                reason="Shared entity",
            ),
        ),
        conflict_candidates=(
            ConflictCandidate(
                first_memory_id="memory-1",
                second_memory_id="memory-2",
                reason="Potential contradiction",
                confidence=0.85,
            ),
        ),
        examined_memory_ids=("memory-1", "memory-2"),
        warnings=("Review candidates before applying changes",),
    )

    decision = JEVDecision(
        action=JEVAction.MERGE,
        reason="Potential duplicate requires review",
        matched_memory_id="memory-1",
        target_memory_ids=("memory-1", "memory-2"),
        confidence=0.97,
        requires_review=True,
        metadata={
            "source": "dream_engine",
            "mutation_applied": False,
        },
    )

    consolidation = ConsolidationResult(decisions=(decision,))

    payload = dream_execution_to_dict(
        result,
        discovery,
        consolidation,
    )

    restored_result, restored_discovery, restored_consolidation = (
        dream_execution_from_dict(payload)
    )

    assert restored_result == result
    assert restored_discovery == discovery
    assert restored_consolidation == consolidation

    assert restored_consolidation.decisions[0].action is JEVAction.MERGE
    assert restored_consolidation.decisions[0].requires_review is True
    assert restored_consolidation.decisions[0].metadata["mutation_applied"] is False

def test_orchestrator_persists_and_retrieves_execution(tmp_path):
    from memory_engine.dreaming import (
        DreamOrchestrator,
        DreamQueue,
        DreamResultStore,
    )
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    queue_path = tmp_path / "queue.sqlite3"
    result_path = tmp_path / "results.sqlite3"
    memory_path = tmp_path / "memories.sqlite3"

    queue = DreamQueue(queue_path)
    result_store = DreamResultStore(result_path)

    service = PersistentMemoryService(
        SQLiteMemoryRepository(memory_path)
    )

    job = queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
        result_store=result_store,
    )

    execution = orchestrator.run_next()

    assert execution is not None
    assert execution.job.status == DreamJobStatus.COMPLETED
    assert result_store.exists(job.job_id)

    # Reopen the stores to simulate application restart.
    reopened_queue = DreamQueue(queue_path)
    reopened_result_store = DreamResultStore(result_path)

    recovered_orchestrator = DreamOrchestrator(
        queue=reopened_queue,
        memory_service=service,
        result_store=reopened_result_store,
    )

    recovered = recovered_orchestrator.get_execution(job.job_id)

    assert recovered.job.status == DreamJobStatus.COMPLETED
    assert recovered.result == execution.result
    assert recovered.discovery == execution.discovery
    assert recovered.consolidation == execution.consolidation


def test_orchestrator_does_not_persist_report_on_discovery_failure(
    tmp_path,
    monkeypatch,
):
    import pytest

    from memory_engine.dreaming import (
        DreamConfig,
        DreamEngine,
        DreamOrchestrator,
        DreamQueue,
        DreamResultStore,
    )
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    queue = DreamQueue(tmp_path / "queue.sqlite3")
    result_store = DreamResultStore(tmp_path / "results.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )

    engine = DreamEngine(DreamConfig(max_attempts=2))
    job = queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
        engine=engine,
        result_store=result_store,
    )

    def fail_discovery(_memories):
        raise RuntimeError("Simulated discovery failure")

    monkeypatch.setattr(
        orchestrator.discovery_engine,
        "discover",
        fail_discovery,
    )

    with pytest.raises(RuntimeError, match="Simulated discovery failure"):
        orchestrator.run_next()

    assert not result_store.exists(job.job_id)
    assert queue.get(job.job_id).status == DreamJobStatus.PENDING