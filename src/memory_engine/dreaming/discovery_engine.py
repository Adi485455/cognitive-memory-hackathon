"""Deterministic discovery of potential memory connections."""

from __future__ import annotations

import re

from memory_engine.dreaming.config import DreamConfig
from memory_engine.dreaming.discovery import (
    ConflictCandidate,
    DiscoveryResult,
    DuplicateCandidate,
    RelationshipCandidate,
)
from memory_engine.kernel.models import Memory, MemoryStatus


class MemoryDiscoveryEngine:
    """Find potential duplicate, related, and conflicting memories."""

    NEGATIONS = {
        "not",
        "never",
        "no",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    }

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been",
        "being", "by", "for", "from", "in", "into", "is",
        "it", "of", "on", "or", "that", "the", "their",
        "this", "to", "was", "were", "with",
    }

    def __init__(self, config: DreamConfig | None = None) -> None:
        self.config = config or DreamConfig()

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        """Normalize text into a set of meaningful tokens."""
        tokens = set(re.findall(r"\b[\w']+\b", text.casefold()))

        return tokens - cls.STOP_WORDS

    @classmethod
    def _similarity(cls, first: str, second: str) -> float:
        """Calculate token-based Jaccard similarity."""
        first_tokens = cls._tokens(first)
        second_tokens = cls._tokens(second)

        if not first_tokens or not second_tokens:
            return 0.0

        return len(first_tokens & second_tokens) / len(
            first_tokens | second_tokens
        )

    @classmethod
    def _shared_entities(cls, first: Memory, second: Memory) -> tuple[str, ...]:
        """Return shared entity IDs."""
        return tuple(
            sorted(
                {str(entity_id) for entity_id in first.entity_ids}
                & {str(entity_id) for entity_id in second.entity_ids}
            )
        )

    @classmethod
    def _possible_conflict(cls, first: str, second: str) -> bool:
        """Detect a narrow lexical signal of possible contradiction.

        This deliberately avoids claiming general natural-language
        contradiction detection.
        """
        first_tokens = cls._tokens(first)
        second_tokens = cls._tokens(second)

        first_negated = bool(first_tokens & cls.NEGATIONS)
        second_negated = bool(second_tokens & cls.NEGATIONS)

        first_claim = first_tokens - cls.NEGATIONS
        second_claim = second_tokens - cls.NEGATIONS

        same_claim = (
            first_claim
            and second_claim
            and first_claim == second_claim
        )

        return bool(same_claim and first_negated != second_negated)

    def discover(
        self,
        memories: tuple[Memory, ...] | list[Memory],
    ) -> DiscoveryResult:
        """Analyze active memories and return bounded candidate lists."""
        eligible = sorted(
            (
                memory
                for memory in memories
                if memory.status == MemoryStatus.ACTIVE
            ),
            key=lambda memory: str(memory.memory_id),
        )

        eligible = eligible[: self.config.max_memories_per_job]

        duplicates: list[DuplicateCandidate] = []
        relationships: list[RelationshipCandidate] = []
        conflicts: list[ConflictCandidate] = []
        warnings: list[str] = []

        duplicate_limit = self.config.max_duplicate_candidates_per_job
        relationship_limit = self.config.max_relationships_per_job
        conflict_limit = self.config.max_conflict_candidates_per_job

        for index, first in enumerate(eligible):
            for second in eligible[index + 1:]:
                first_id = str(first.memory_id)
                second_id = str(second.memory_id)

                # Keep candidate generation deterministic.
                similarity = self._similarity(
                    first.content,
                    second.content,
                )

                if (
                    similarity >= 0.8
                    and len(duplicates) < duplicate_limit
                ):
                    duplicates.append(
                        DuplicateCandidate(
                            first_memory_id=first_id,
                            second_memory_id=second_id,
                            similarity=similarity,
                            reason="High token-based Jaccard similarity",
                        )
                    )

                shared_entities = self._shared_entities(first, second)

                if (
                    shared_entities
                    and len(relationships) < relationship_limit
                ):
                    relationships.append(
                        RelationshipCandidate(
                            first_memory_id=first_id,
                            second_memory_id=second_id,
                            relation_type="shares_entity",
                            confidence=0.8,
                            reason=(
                                "Shared entity IDs: "
                                + ", ".join(shared_entities)
                            ),
                        )
                    )

                if (
                    self._possible_conflict(
                        first.content,
                        second.content,
                    )
                    and len(conflicts) < conflict_limit
                ):
                    conflicts.append(
                        ConflictCandidate(
                            first_memory_id=first_id,
                            second_memory_id=second_id,
                            reason=(
                                "Matching token sets with differing "
                                "negation signals"
                            ),
                            confidence=0.6,
                        )
                    )

        if len(memories) > len(eligible):
            warnings.append(
                "Input was filtered or truncated to the configured "
                "active-memory limit."
            )

        return DiscoveryResult(
            duplicate_candidates=tuple(duplicates),
            relationship_candidates=tuple(relationships),
            conflict_candidates=tuple(conflicts),
            examined_memory_ids=tuple(
                str(memory.memory_id) for memory in eligible
            ),
            warnings=tuple(warnings),
        )