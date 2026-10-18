"""Convert discovery candidates into safe JEV decisions."""

from __future__ import annotations

from dataclasses import dataclass

from memory_engine.decision.jev import JEVAction, JEVDecision
from memory_engine.dreaming.config import DreamConfig
from memory_engine.dreaming.discovery import DiscoveryResult


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Decisions proposed for a single dreaming pass."""

    decisions: tuple[JEVDecision, ...] = ()

    @property
    def requires_review(self) -> tuple[JEVDecision, ...]:
        """Return decisions requiring review."""
        return tuple(
            decision
            for decision in self.decisions
            if decision.requires_review
        )

    @property
    def automatic_safe_actions(self) -> tuple[JEVDecision, ...]:
        """Return non-mutating decisions that do not require review."""
        return tuple(
            decision
            for decision in self.decisions
            if not decision.requires_review
            and decision.action == JEVAction.IGNORE
        )


class ConsolidationDecisionEngine:
    """Produce conservative decisions without mutating canonical memories."""

    def __init__(
        self,
        config: DreamConfig | None = None,
        *,
        minimum_duplicate_similarity: float = 0.95,
        minimum_relationship_confidence: float = 0.8,
        minimum_conflict_confidence: float = 0.8,
    ) -> None:
        self.config = config or DreamConfig()

        for name, value in (
            ("minimum_duplicate_similarity", minimum_duplicate_similarity),
            (
                "minimum_relationship_confidence",
                minimum_relationship_confidence,
            ),
            ("minimum_conflict_confidence", minimum_conflict_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        self.minimum_duplicate_similarity = minimum_duplicate_similarity
        self.minimum_relationship_confidence = (
            minimum_relationship_confidence
        )
        self.minimum_conflict_confidence = minimum_conflict_confidence

    def evaluate(
        self,
        discovery: DiscoveryResult,
    ) -> ConsolidationResult:
        """Turn bounded discovery output into reviewable JEV decisions."""
        decisions: list[JEVDecision] = []

        duplicates = discovery.duplicate_candidates[
            : self.config.max_duplicate_candidates_per_job
        ]
        relationships = discovery.relationship_candidates[
            : self.config.max_relationships_per_job
        ]
        conflicts = discovery.conflict_candidates[
            : self.config.max_conflict_candidates_per_job
        ]

        for candidate in duplicates:
            targets = (
                candidate.first_memory_id,
                candidate.second_memory_id,
            )

            if candidate.similarity >= self.minimum_duplicate_similarity:
                decisions.append(
                    JEVDecision(
                        action=JEVAction.MERGE,
                        reason=(
                            "Potential duplicate requires validation before "
                            "any consolidation."
                        ),
                        matched_memory_id=candidate.first_memory_id,
                        target_memory_ids=targets,
                        confidence=candidate.similarity,
                        requires_review=True,
                        metadata={
                            "source": "dream_engine",
                            "candidate_type": "duplicate",
                            "similarity": candidate.similarity,
                            "discovery_reason": candidate.reason,
                            "mutation_applied": False,
                        },
                    )
                )
            else:
                decisions.append(
                    JEVDecision(
                        action=JEVAction.UNCERTAIN,
                        reason=(
                            "Duplicate similarity is insufficient for a "
                            "consolidation proposal."
                        ),
                        target_memory_ids=targets,
                        confidence=candidate.similarity,
                        requires_review=True,
                        metadata={
                            "source": "dream_engine",
                            "candidate_type": "duplicate",
                            "mutation_applied": False,
                        },
                    )
                )

        for candidate in relationships:
            targets = (
                candidate.first_memory_id,
                candidate.second_memory_id,
            )

            if (
                candidate.confidence
                >= self.minimum_relationship_confidence
            ):
                decisions.append(
                    JEVDecision(
                        action=JEVAction.CREATE_RELATION,
                        reason=(
                            "Potential relationship requires validation "
                            "before it is persisted."
                        ),
                        matched_memory_id=candidate.first_memory_id,
                        target_memory_ids=targets,
                        confidence=candidate.confidence,
                        requires_review=True,
                        metadata={
                            "source": "dream_engine",
                            "candidate_type": "relationship",
                            "relation_type": candidate.relation_type,
                            "discovery_reason": candidate.reason,
                            "mutation_applied": False,
                        },
                    )
                )
            else:
                decisions.append(
                    JEVDecision(
                        action=JEVAction.UNCERTAIN,
                        reason=(
                            "Relationship confidence is below the "
                            "configured threshold."
                        ),
                        target_memory_ids=targets,
                        confidence=candidate.confidence,
                        requires_review=True,
                        metadata={
                            "source": "dream_engine",
                            "candidate_type": "relationship",
                            "mutation_applied": False,
                        },
                    )
                )

        for candidate in conflicts:
            targets = (
                candidate.first_memory_id,
                candidate.second_memory_id,
            )

            if candidate.confidence >= self.minimum_conflict_confidence:
                action = JEVAction.CONFLICT
                reason = (
                    "Potential conflict requires investigation; neither "
                    "memory will be changed automatically."
                )
            else:
                action = JEVAction.UNCERTAIN
                reason = (
                    "Potential conflict has insufficient confidence "
                    "for a stronger conclusion."
                )

            decisions.append(
                JEVDecision(
                    action=action,
                    reason=reason,
                    matched_memory_id=candidate.first_memory_id,
                    target_memory_ids=targets,
                    confidence=candidate.confidence,
                    requires_review=True,
                    metadata={
                        "source": "dream_engine",
                        "candidate_type": "conflict",
                        "discovery_reason": candidate.reason,
                        "mutation_applied": False,
                    },
                )
            )

        return ConsolidationResult(decisions=tuple(decisions))