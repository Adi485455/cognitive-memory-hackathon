"""TypeSafe JEV provider adapter.

External judgments are converted into internal JEVDecision objects.
The adapter never mutates canonical memories.
"""

from typing import Any

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from memory_engine.decision.jev import JEVAction, JEVDecision
from memory_engine.formation.candidate import CandidateMemory
from memory_engine.kernel.models import Memory


class TypeSafeJEVAdapter:
    """Request structured memory decisions from TypeSafe JEV."""

    ACTIONS = {
        action.value.upper(): action
        for action in JEVAction
        if action not in {JEVAction.ARCHIVE, JEVAction.DECAY}
    }

    def __init__(self, model: str = "jev-latest") -> None:
        self.model = model

    @staticmethod
    def _confidence_from_score(value: Any) -> float:
        """Convert the provider's qualitative score into a conservative value."""
        label = str(value).strip().casefold()

        confidence_map = {
            "weak evidence": 0.35,
            "moderate evidence": 0.65,
            "strong evidence": 0.9,
        }

        return confidence_map.get(label, 0.5)

    def evaluate(
        self,
        candidate: CandidateMemory,
        existing_memories: tuple[Memory, ...],
    ) -> JEVDecision:
        """Ask TypeSafe JEV to classify a candidate.

        Raises provider exceptions to the caller. Callers must not interpret
        a failed request as permission to store a candidate.
        """
        state = {
            "candidate": {
                "content": candidate.content,
                "memory_type": candidate.memory_type.value,
                "scope": candidate.scope,
                "importance": candidate.importance,
                "confidence": candidate.confidence,
                "source_event_id": candidate.source_event_id,
            },
            "existing_memories": [
                {
                    "memory_id": str(memory.memory_id),
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "scope": memory.scope,
                    "status": memory.status.value,
                    "version": memory.version,
                }
                for memory in existing_memories
            ],
        }

        action_criteria = {
            "STORE": "Store this as a new memory.",
            "IGNORE": "Do not store because it is redundant or irrelevant.",
            "UPDATE": "Correct an existing memory using new information.",
            "MERGE": "Combine compatible existing memories.",
            "SUPERSEDE": "Replace an outdated memory while preserving history.",
            "REINFORCE": "Attach new supporting evidence to an existing memory.",
            "CONFLICT": "The candidate contradicts existing information.",
            "CREATE_RELATION": "Create a meaningful relation between entities or memories.",
            "UNCERTAIN": "There is insufficient evidence for a safe decision.",
            "ESCALATE": "Further reasoning or human review is required.",
        }

        with TypeSafeClient(model=self.model) as client:
            response = client.system_one(
                state=state,
                questions={
                    "action": Choice(
                        instructions=(
                            "Choose the safest appropriate memory action. "
                            "Respect scope, memory status, provenance, and "
                            "temporal changes. Never invent facts."
                        ),
                        criteria=action_criteria,
                    ),
                    "confidence": Score(
                        instructions=(
                            "Rate the strength of the evidence supporting "
                            "the proposed action."
                        ),
                        criteria=[
                            "Weak evidence",
                            "Moderate evidence",
                            "Strong evidence",
                        ],
                    ),
                    "needs_review": Noul(
                        instructions=(
                            "Does this decision require additional reasoning "
                            "or human review before canonical memory changes?"
                        ),
                    ),
                },
            )

        raw_action = str(response.choices["action"].choice).strip().upper()
        action = self.ACTIONS.get(raw_action)

        if action is None:
            return JEVDecision(
                action=JEVAction.UNCERTAIN,
                reason=f"Unsupported JEV action returned: {raw_action!r}",
                confidence=0.0,
                requires_review=True,
                metadata={
                    "provider": "typesafe",
                    "model": self.model,
                    "raw_action": raw_action,
                },
            )

        raw_score = response.scores["confidence"].score
        confidence = self._confidence_from_score(raw_score)

        raw_review = response.nouls["needs_review"].noul
        try:
            review_probability = float(raw_review)
        except (TypeError, ValueError):
            review_probability = 1.0

        if not 0.0 <= review_probability <= 1.0:
            review_probability = 1.0

        requires_review = (
            review_probability >= 0.5
            or action in {
                JEVAction.UPDATE,
                JEVAction.MERGE,
                JEVAction.SUPERSEDE,
                JEVAction.CONFLICT,
                JEVAction.CREATE_RELATION,
                JEVAction.ESCALATE,
                JEVAction.UNCERTAIN,
            }
        )

        # The provider classifies the action but does not supply a separately
        # validated target-selection contract. Do not invent target IDs.
        target_required = action in {
            JEVAction.UPDATE,
            JEVAction.MERGE,
            JEVAction.SUPERSEDE,
            JEVAction.REINFORCE,
            JEVAction.CONFLICT,
            JEVAction.CREATE_RELATION,
        }

        if target_required:
            requires_review = True

        return JEVDecision(
            action=action,
            reason="Decision returned by TypeSafe JEV.",
            confidence=confidence,
            requires_review=requires_review,
            metadata={
                "provider": "typesafe",
                "model": self.model,
                "provider_score": str(raw_score),
                "review_probability": review_probability,
                "target_selection_required": target_required,
            },
        )