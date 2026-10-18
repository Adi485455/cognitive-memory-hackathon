"""Coordinate experience ingestion, decisions, and canonical storage."""

from dataclasses import dataclass
from typing import Any

from memory_engine.decision.jev import JEV, JEVAction, JEVDecision
from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from memory_engine.kernel.ids import MemoryId
from memory_engine.kernel.models import Memory
from memory_engine.storage.evidence_registry import EvidenceRegistry
from memory_engine.storage.persistent_memory_service import (
    PersistentMemoryService,
)


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Outcome of processing one incoming experience."""

    event_id: str
    action: str
    reason: str
    confidence: float
    requires_review: bool
    memory: Memory | None = None
    evidence_id: str | None = None


class MemoryOrchestrator:
    """Coordinate formation, JEV evaluation, and canonical persistence."""

    def __init__(
        self,
        memory_service: PersistentMemoryService,
        evidence_registry: EvidenceRegistry,
        ingestor: ExperienceIngestor | None = None,
        candidate_builder: CandidateBuilder | None = None,
        jev: JEV | None = None,
    ) -> None:
        self.memory_service = memory_service
        self.evidence_registry = evidence_registry
        self.ingestor = ingestor or ExperienceIngestor()
        self.candidate_builder = candidate_builder or CandidateBuilder()
        self.jev = jev or JEV()

    def process(
        self,
        content: str,
        *,
        source: str = "conversation",
        scope: str = "user",
        memory_type=None,
        importance: float = 0.5,
        confidence: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> OrchestrationResult:
        """Process an experience and store it only when JEV permits."""

        from memory_engine.kernel.models import MemoryType

        selected_type = memory_type or MemoryType.SEMANTIC

        event = self.ingestor.ingest(
            content,
            source=source,
            scope=scope,
            metadata=metadata,
        )

        candidate = self.candidate_builder.build(
            event,
            memory_type=selected_type,
            importance=importance,
            confidence=confidence,
        )

        existing_memories = tuple(
            memory
            for memory in self.memory_service.list_memories()
            if memory.scope == candidate.scope
        )

        decision: JEVDecision = self.jev.evaluate(
            candidate,
            existing_memories,
        )

        # A recommendation requiring review must never mutate canonical state.
        if decision.requires_review or decision.action != JEVAction.STORE:
            return OrchestrationResult(
                event_id=event.event_id,
                action=decision.action.value,
                reason=decision.reason,
                confidence=decision.confidence,
                requires_review=decision.requires_review,
            )

        evidence = self.evidence_registry.register(
            candidate.content.encode("utf-8"),
            source=candidate.source,
            metadata={
                **candidate.metadata,
                "scope": candidate.scope,
                "source_event_id": candidate.source_event_id,
            },
        )

        memory = Memory(
            memory_id=MemoryId.new(),
            content=candidate.content,
            memory_type=candidate.memory_type,
            evidence_ids=(evidence.evidence_id,),
            importance=candidate.importance,
            confidence=candidate.confidence,
            scope=candidate.scope,
            metadata={
                **candidate.metadata,
                "source": candidate.source,
                "source_event_id": candidate.source_event_id,
                "evidence_content_hash": evidence.content_hash,
            },
        )

        stored_memory = self.memory_service.create(memory)

        return OrchestrationResult(
            event_id=event.event_id,
            action=decision.action.value,
            reason=decision.reason,
            confidence=decision.confidence,
            requires_review=False,
            memory=stored_memory,
            evidence_id=str(evidence.evidence_id),
        )