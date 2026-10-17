"""Structured decision contract for memory formation and evolution.

JEV recommends an action. The Memory Manager remains responsible for
validating and applying mutations to canonical memory.
"""

from dataclasses import dataclass
from enum import Enum

from memory_engine.formation.candidate import CandidateMemory
from memory_engine.kernel.models import Memory, MemoryStatus


class JEVAction(str, Enum):
    """Supported memory decision types."""

    STORE = "store"
    IGNORE = "ignore"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    REINFORCE = "reinforce"
    CONFLICT = "conflict"
    CREATE_RELATION = "create_relation"
    UNCERTAIN = "uncertain"
    ESCALATE = "escalate"

    # Memory evolution actions.
    ARCHIVE = "archive"
    DECAY = "decay"


@dataclass(frozen=True, slots=True)
class JEVDecision:
    """A structured recommendation for a memory candidate.

    Attributes:
        action: Recommended operation.
        reason: Human-readable explanation.
        matched_memory_id: Primary target, if applicable.
        target_memory_ids: All existing memories involved in the decision.
        confidence: Confidence in the recommendation, between 0 and 1.
        requires_review: Whether further reasoning or human review is needed.
        metadata: Additional structured decision information.
    """

    action: JEVAction
    reason: str
    matched_memory_id: str | None = None
    target_memory_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    requires_review: bool = False
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Decision reason cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Decision confidence must be between 0 and 1")

        if (
            self.matched_memory_id is not None
            and not self.matched_memory_id.strip()
        ):
            raise ValueError("Matched memory ID cannot be empty")

        if any(not memory_id.strip() for memory_id in self.target_memory_ids):
            raise ValueError("Target memory IDs cannot be empty")

        if (
            self.matched_memory_id is not None
            and self.target_memory_ids
            and self.matched_memory_id not in self.target_memory_ids
        ):
            raise ValueError(
                "Matched memory ID must be included in target memory IDs"
            )

        # Avoid exposing mutable metadata owned by the caller.
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class JEV:
    """Apply conservative deterministic rules to memory candidates.

    This initial implementation handles low-confidence candidates and exact
    duplicate detection. Other decisions are represented by the contract but
    require additional reasoning and validated execution paths.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.3

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for conservative exact-match detection."""
        return " ".join(text.casefold().split())

    def evaluate(
        self,
        candidate: CandidateMemory,
        existing_memories: tuple[Memory, ...],
    ) -> JEVDecision:
        """Recommend an action without mutating canonical memory."""

        if candidate.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            return JEVDecision(
                action=JEVAction.UNCERTAIN,
                reason="Candidate confidence is too low for automatic storage.",
                confidence=candidate.confidence,
                requires_review=True,
                metadata={"rule": "low_candidate_confidence"},
            )

        normalized_candidate = self._normalize(candidate.content)

        for memory in existing_memories:
            if memory.status != MemoryStatus.ACTIVE:
                continue

            if memory.scope != candidate.scope:
                continue

            if self._normalize(memory.content) == normalized_candidate:
                memory_id = str(memory.memory_id)

                return JEVDecision(
                    action=JEVAction.IGNORE,
                    reason="An equivalent active memory already exists.",
                    matched_memory_id=memory_id,
                    target_memory_ids=(memory_id,),
                    confidence=1.0,
                    metadata={"rule": "normalized_exact_match"},
                )

        return JEVDecision(
            action=JEVAction.STORE,
            reason="No equivalent active memory was found.",
            confidence=candidate.confidence,
            metadata={"rule": "no_active_exact_match"},
        )