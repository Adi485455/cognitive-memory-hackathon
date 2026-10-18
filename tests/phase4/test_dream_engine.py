import pytest

from datetime import datetime, timezone

from memory_engine.dreaming import DreamCandidateSelector
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.models import (
    Memory,
    MemoryStatus,
    MemoryType,
)

from memory_engine.dreaming import (
    DreamConfig,
    DreamEngine,
    DreamJob,
    DreamJobStatus,
    DreamResult,
)
from uuid import uuid5, NAMESPACE_DNS
from memory_engine.dreaming import MemoryDiscoveryEngine

from memory_engine.decision.jev import JEVAction
from datetime import datetime, timezone

from memory_engine.dreaming import (
    ConflictCandidate,
    ConsolidationDecisionEngine,
    ConsolidationResult,
    DiscoveryResult,
    DuplicateCandidate,
    DreamCandidateSelector,
    DreamConfig,
    DreamEngine,
    DreamJob,
    DreamJobStatus,
    DreamQueue,
    DreamResult,
    MemoryDiscoveryEngine,
    RelationshipCandidate,
)

from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.models import (
    Memory,
    MemoryStatus,
    MemoryType,
)
from memory_engine.dreaming import DreamOrchestrator
from memory_engine.storage.persistent_memory_service import (
    PersistentMemoryService,
)
from memory_engine.storage.sqlite_memory_repository import (
    SQLiteMemoryRepository,
)


def test_engine_starts_pending_job():
    engine = DreamEngine()
    job = DreamJob()

    started = engine.start_job(job)

    assert started.status == DreamJobStatus.RUNNING
    assert started.started_at is not None
    assert started.attempt == 1

    # The original immutable job remains unchanged.
    assert job.status == DreamJobStatus.PENDING


def test_engine_rejects_starting_non_pending_job():
    engine = DreamEngine()
    job = DreamJob()
    started = engine.start_job(job)

    with pytest.raises(ValueError, match="pending"):
        engine.start_job(started)


def test_engine_enforces_maximum_attempts():
    engine = DreamEngine(DreamConfig(max_attempts=2))
    job = DreamJob()

    first = engine.start_job(job)
    failed = engine.fail_job(first, "temporary failure")

    # Retry by constructing a new pending attempt from the failed job.
    retry = DreamJob(
        job_id=failed.job_id,
        target_memory_ids=failed.target_memory_ids,
        trigger_event_ids=failed.trigger_event_ids,
        attempt=failed.attempt,
    )
    second = engine.start_job(retry)

    assert second.attempt == 2

    with pytest.raises(ValueError, match="Maximum"):
        engine.start_job(second)


def test_engine_completes_matching_running_job():
    engine = DreamEngine()
    started = engine.start_job(DreamJob())
    result = DreamResult(job_id=started.job_id)

    completed = engine.complete_job(started, result)

    assert completed.status == DreamJobStatus.COMPLETED
    assert completed.completed_at is not None


def test_engine_rejects_result_for_another_job():
    engine = DreamEngine()
    started = engine.start_job(DreamJob())
    result = DreamResult(job_id="different-job")

    with pytest.raises(ValueError, match="does not match"):
        engine.complete_job(started, result)


def test_engine_fails_running_job():
    engine = DreamEngine()
    started = engine.start_job(DreamJob())

    failed = engine.fail_job(started, "analysis failed")

    assert failed.status == DreamJobStatus.FAILED
    assert failed.error == "analysis failed"


def test_engine_rejects_empty_failure_reason():
    engine = DreamEngine()
    started = engine.start_job(DreamJob())

    with pytest.raises(ValueError, match="empty"):
        engine.fail_job(started, "   ")


def test_dream_queue_persists_jobs_after_restart(tmp_path):
    from memory_engine.dreaming import DreamQueue

    database = tmp_path / "dream_queue.sqlite3"
    queue = DreamQueue(database)

    job = DreamJob(trigger_event_ids=("event-1",))
    queue.enqueue(job)

    reopened = DreamQueue(database)

    assert reopened.get(job.job_id) == job


def test_dream_queue_deduplicates_trigger_events(tmp_path):
    from memory_engine.dreaming import DreamQueue

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")

    first = DreamJob(trigger_event_ids=("event-1", "event-2"))
    second = DreamJob(trigger_event_ids=("event-2", "event-1"))

    queue.enqueue(first)
    returned = queue.enqueue(second)

    assert returned.job_id == first.job_id
    assert len(queue.list_jobs()) == 1


def test_dream_queue_claims_pending_job(tmp_path):
    from memory_engine.dreaming import DreamQueue

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    engine = DreamEngine()
    job = DreamJob()

    queue.enqueue(job)
    claimed = queue.claim_next(engine)

    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == DreamJobStatus.RUNNING
    assert claimed.attempt == 1
    assert queue.claim_next(engine) is None


def test_dream_queue_saves_completed_job(tmp_path):
    from memory_engine.dreaming import DreamQueue

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    engine = DreamEngine()
    job = queue.enqueue(DreamJob())

    claimed = queue.claim_next(engine)
    assert claimed is not None

    result = DreamResult(job_id=claimed.job_id)
    completed = engine.complete_job(claimed, result)

    queue.save(completed)

    assert queue.get(job.job_id).status == DreamJobStatus.COMPLETED


def test_dream_queue_rejects_non_pending_job(tmp_path):
    from memory_engine.dreaming import DreamQueue

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    running = DreamEngine().start_job(DreamJob())

    with pytest.raises(ValueError, match="pending"):
        queue.enqueue(running)
        from uuid import uuid4


def make_candidate_memory(
    memory_id: str,
    *,
    importance: float = 0.5,
    confidence: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    """Create a valid memory for candidate-selection tests."""
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Convert readable test labels into valid, deterministic UUIDs.
    valid_memory_id = MemoryId(
        str(
            uuid5(
                NAMESPACE_DNS,
                f"candidate-test:{memory_id}",
            )
        )
    )

    return Memory(
        memory_id=valid_memory_id,
        content=f"Memory content {memory_id}",
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(EvidenceId.new(),),
        created_at=timestamp,
        updated_at=timestamp,
        learned_at=timestamp,
        status=status,
        importance=importance,
        confidence=confidence,
    )



def test_candidate_selector_excludes_inactive_memories():
    selector = DreamCandidateSelector()

    active = make_candidate_memory("active-memory")
    archived = make_candidate_memory(
        "archived-memory",
        status=MemoryStatus.ARCHIVED,
    )
    superseded = make_candidate_memory(
        "superseded-memory",
        status=MemoryStatus.SUPERSEDED,
    )

    selected = selector.select(
        [active, archived, superseded],
        DreamJob(),
    )

    assert selected == (active,)


def test_candidate_selector_respects_explicit_targets():
    selector = DreamCandidateSelector()

    first = make_candidate_memory("memory-one")
    second = make_candidate_memory("memory-two")
    third = make_candidate_memory("memory-three")

    job = DreamJob(
        target_memory_ids=(
            str(second.memory_id),
            str(third.memory_id),
        )
    )

    selected = selector.select([first, second, third], job)

    assert {str(memory.memory_id) for memory in selected} == {
        str(second.memory_id),
        str(third.memory_id),
    }


def test_candidate_selector_prioritizes_importance():
    selector = DreamCandidateSelector()

    low = make_candidate_memory(
        "low",
        importance=0.2,
        confidence=0.9,
    )
    high = make_candidate_memory(
        "high",
        importance=0.9,
        confidence=0.5,
    )
    medium = make_candidate_memory(
        "medium",
        importance=0.5,
        confidence=0.8,
    )

    selected = selector.select(
        [low, high, medium],
        DreamJob(),
    )

    assert selected == (high, medium, low)


def test_candidate_selector_respects_job_limit():
    selector = DreamCandidateSelector(
        DreamConfig(max_memories_per_job=2)
    )

    memories = [
        make_candidate_memory(
            f"memory-{index}",
            importance=index / 10,
        )
        for index in range(5)
    ]

    selected = selector.select(memories, DreamJob())

    assert len(selected) == 2
    assert selected[0].importance == 0.4
    assert selected[1].importance == 0.3


def test_candidate_selector_does_not_mutate_memories():
    selector = DreamCandidateSelector()
    memory = make_candidate_memory(
        "unchanged",
        importance=0.7,
    )

    original_version = memory.version
    original_content = memory.content

    selector.select([memory], DreamJob())

    assert memory.version == original_version
    assert memory.content == original_content

    from memory_engine.dreaming import MemoryDiscoveryEngine


def test_discovery_finds_duplicate_candidates():
    engine = MemoryDiscoveryEngine()

    first = make_candidate_memory("duplicate-one")
    second = make_candidate_memory("duplicate-two")

    # Use equivalent content to guarantee high token similarity.
    from dataclasses import replace

    first = replace(
        first,
        content="Aditya prefers Python programming for backend development",
    )
    second = replace(
        second,
        content="Aditya prefers Python programming for backend development",
    )

    result = engine.discover([first, second])

    assert len(result.duplicate_candidates) == 1
    assert result.duplicate_candidates[0].similarity == 1.0


def test_discovery_finds_shared_entity_relationship():
    from dataclasses import replace

    from memory_engine.kernel.ids import EntityId

    engine = MemoryDiscoveryEngine()
    shared_entity = EntityId.new()

    first = replace(
        make_candidate_memory("relationship-one"),
        entity_ids=(shared_entity,),
    )
    second = replace(
        make_candidate_memory("relationship-two"),
        entity_ids=(shared_entity,),
    )

    result = engine.discover([first, second])

    assert len(result.relationship_candidates) == 1
    assert result.relationship_candidates[0].relation_type == "shares_entity"


def test_discovery_detects_simple_negation_conflict():
    from dataclasses import replace

    engine = MemoryDiscoveryEngine()

    first = replace(
        make_candidate_memory("conflict-one"),
        content="Project Apollo is active",
    )
    second = replace(
        make_candidate_memory("conflict-two"),
        content="Project Apollo is not active",
    )

    result = engine.discover([first, second])

    assert len(result.conflict_candidates) == 1


def test_discovery_respects_candidate_limits():
    from dataclasses import replace

    engine = MemoryDiscoveryEngine(
        DreamConfig(
            max_memories_per_job=10,
            max_duplicate_candidates_per_job=1,
        )
    )

    memories = [
        replace(
            make_candidate_memory(f"duplicate-limit-{index}"),
            content="The same repeated memory content",
        )
        for index in range(4)
    ]

    result = engine.discover(memories)

    assert len(result.duplicate_candidates) == 1


def test_discovery_does_not_mutate_canonical_memories():
    from dataclasses import replace

    engine = MemoryDiscoveryEngine()
    first = make_candidate_memory("safe-one")
    second = replace(
        make_candidate_memory("safe-two"),
        content=first.content,
    )

    original_first = first
    original_second = second

    engine.discover([first, second])

    assert first == original_first
    assert second == original_second

    from memory_engine.dreaming import (
    ConflictCandidate,
    ConsolidationDecisionEngine,
    DiscoveryResult,
    DuplicateCandidate,
    RelationshipCandidate,
)
from memory_engine.decision.jev import JEVAction


def test_consolidation_proposes_duplicate_review():
    engine = ConsolidationDecisionEngine()

    discovery = DiscoveryResult(
        duplicate_candidates=(
            DuplicateCandidate(
                first_memory_id="memory-a",
                second_memory_id="memory-b",
                similarity=0.98,
                reason="High token similarity",
            ),
        )
    )

    result = engine.evaluate(discovery)

    assert len(result.decisions) == 1

    decision = result.decisions[0]
    assert decision.action == JEVAction.MERGE
    assert decision.requires_review is True
    assert decision.target_memory_ids == ("memory-a", "memory-b")
    assert decision.metadata["mutation_applied"] is False


def test_consolidation_marks_low_similarity_uncertain():
    engine = ConsolidationDecisionEngine()

    discovery = DiscoveryResult(
        duplicate_candidates=(
            DuplicateCandidate(
                first_memory_id="memory-a",
                second_memory_id="memory-b",
                similarity=0.82,
                reason="Moderate token similarity",
            ),
        )
    )

    result = engine.evaluate(discovery)

    assert result.decisions[0].action == JEVAction.UNCERTAIN
    assert result.decisions[0].requires_review is True


def test_consolidation_proposes_relationship_for_review():
    engine = ConsolidationDecisionEngine()

    discovery = DiscoveryResult(
        relationship_candidates=(
            RelationshipCandidate(
                first_memory_id="memory-a",
                second_memory_id="memory-b",
                relation_type="shares_entity",
                confidence=0.9,
                reason="Shared entity",
            ),
        )
    )

    result = engine.evaluate(discovery)

    decision = result.decisions[0]
    assert decision.action == JEVAction.CREATE_RELATION
    assert decision.requires_review is True
    assert decision.metadata["mutation_applied"] is False


def test_consolidation_escalates_conflict():
    engine = ConsolidationDecisionEngine()

    discovery = DiscoveryResult(
        conflict_candidates=(
            ConflictCandidate(
                first_memory_id="memory-a",
                second_memory_id="memory-b",
                reason="Possible contradictory claims",
                confidence=0.9,
            ),
        )
    )

    result = engine.evaluate(discovery)

    decision = result.decisions[0]
    assert decision.action == JEVAction.CONFLICT
    assert decision.requires_review is True
    assert len(result.requires_review) == 1


def test_consolidation_respects_candidate_limits():
    engine = ConsolidationDecisionEngine(
        DreamConfig(
            max_duplicate_candidates_per_job=1,
        )
    )

    discovery = DiscoveryResult(
        duplicate_candidates=(
            DuplicateCandidate("a", "b", 0.99, "First pair"),
            DuplicateCandidate("c", "d", 0.99, "Second pair"),
        )
    )

    result = engine.evaluate(discovery)

    assert len(result.decisions) == 1


def test_consolidation_does_not_auto_apply_mutations():
    engine = ConsolidationDecisionEngine()

    discovery = DiscoveryResult(
        duplicate_candidates=(
            DuplicateCandidate(
                first_memory_id="memory-a",
                second_memory_id="memory-b",
                similarity=1.0,
                reason="Exact content match",
            ),
        )
    )

    result = engine.evaluate(discovery)

    assert result.decisions[0].action == JEVAction.MERGE
    assert result.decisions[0].requires_review is True
    assert result.decisions[0].metadata["mutation_applied"] is False

def test_orchestrator_returns_none_for_empty_queue(tmp_path):
    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
    )

    assert orchestrator.run_next() is None


def test_orchestrator_completes_job_and_persists_state(tmp_path):
    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )

    memory = make_candidate_memory("orchestrator-memory")
    service.create(memory)

    job = queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
    )

    execution = orchestrator.run_next()

    assert execution is not None
    assert execution.job.job_id == job.job_id
    assert execution.job.status == DreamJobStatus.COMPLETED
    assert execution.result.job_id == job.job_id
    assert execution.result.mutations_applied == 0
    assert (
        queue.get(job.job_id).status
        == DreamJobStatus.COMPLETED
    )


def test_orchestrator_does_not_mutate_canonical_memories(tmp_path):
    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )

    first = make_candidate_memory("unchanged-first")
    second = make_candidate_memory("unchanged-second")

    service.create(first)
    service.create(second)

    before = service.list_memories()
    queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
    )

    execution = orchestrator.run_next()

    assert execution is not None
    assert service.list_memories() == before
    assert execution.result.mutations_applied == 0

def test_orchestrator_retries_transient_failure(
    tmp_path,
    monkeypatch,
):
        
    from memory_engine.dreaming import DreamOrchestrator
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )
    engine = DreamEngine(DreamConfig(max_attempts=3))

    job = queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
        engine=engine,
    )

    calls = 0
    original_discover = orchestrator.discovery_engine.discover

    def fail_once_then_succeed(memories):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise RuntimeError("Temporary discovery failure")

        return original_discover(memories)

    monkeypatch.setattr(
        orchestrator.discovery_engine,
        "discover",
        fail_once_then_succeed,
    )

    with pytest.raises(RuntimeError, match="Temporary discovery failure"):
        orchestrator.run_next()

    after_failure = queue.get(job.job_id)

    assert after_failure.status == DreamJobStatus.PENDING
    assert after_failure.attempt == 1
    assert after_failure.error == "Temporary discovery failure"

    execution = orchestrator.run_next()

    assert execution is not None
    assert execution.job.job_id == job.job_id
    assert execution.job.status == DreamJobStatus.COMPLETED
    assert execution.job.attempt == 2
    assert calls == 2


def test_orchestrator_marks_job_failed_after_retry_limit(
    tmp_path,
    monkeypatch,
):
    from memory_engine.dreaming import DreamOrchestrator
    from memory_engine.storage.persistent_memory_service import (
        PersistentMemoryService,
    )
    from memory_engine.storage.sqlite_memory_repository import (
        SQLiteMemoryRepository,
    )

    queue = DreamQueue(tmp_path / "dream_queue.sqlite3")
    service = PersistentMemoryService(
        SQLiteMemoryRepository(tmp_path / "memories.sqlite3")
    )
    engine = DreamEngine(DreamConfig(max_attempts=2))

    job = queue.enqueue(DreamJob())

    orchestrator = DreamOrchestrator(
        queue=queue,
        memory_service=service,
        engine=engine,
    )

    def always_fail(_memories):
        raise RuntimeError("Discovery unavailable")

    monkeypatch.setattr(
        orchestrator.discovery_engine,
        "discover",
        always_fail,
    )

    # First attempt fails and is requeued.
    with pytest.raises(RuntimeError, match="Discovery unavailable"):
        orchestrator.run_next()

    first_failure = queue.get(job.job_id)

    assert first_failure.status == DreamJobStatus.PENDING
    assert first_failure.attempt == 1

    # Second attempt reaches the configured maximum.
    with pytest.raises(RuntimeError, match="Discovery unavailable"):
        orchestrator.run_next()

    final_job = queue.get(job.job_id)

    assert final_job.status == DreamJobStatus.FAILED
    assert final_job.attempt == 2
    assert final_job.error == "Discovery unavailable"

    # No further attempt should be available.
    assert orchestrator.run_next() is None

    