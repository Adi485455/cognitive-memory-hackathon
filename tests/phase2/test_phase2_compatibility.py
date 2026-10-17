from memory_engine.decision.jev import JEV, JEVAction
from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.manager import MemoryManager
from memory_engine.kernel.models import Memory, MemoryType


def test_formation_and_jev_work_with_memory_kernel():
    ingestor = ExperienceIngestor()
    builder = CandidateBuilder()
    manager = MemoryManager()
    jev = JEV()

    event = ingestor.ingest(
        "The cognitive memory project uses a canonical memory model",
        source="conversation",
        scope="project",
    )

    candidate = builder.build(
        event,
        memory_type=MemoryType.SEMANTIC,
        confidence=0.9,
    )

    decision = jev.evaluate(candidate, ())

    assert decision.action == JEVAction.STORE

    memory = Memory(
        memory_id=MemoryId.new(),
        content=candidate.content,
        memory_type=candidate.memory_type,
        evidence_ids=(EvidenceId.new(),),
        scope=candidate.scope,
        importance=candidate.importance,
        confidence=candidate.confidence,
        metadata={
            **candidate.metadata,
            "source": candidate.source,
            "source_event_id": candidate.source_event_id,
        },
    )

    stored_memory = manager.create(memory)

    assert stored_memory.content == candidate.content
    assert stored_memory.scope == event.scope
    assert stored_memory.evidence_ids == memory.evidence_ids

    duplicate_decision = jev.evaluate(
        candidate,
        manager.list_memories(),
    )

    assert duplicate_decision.action == JEVAction.IGNORE