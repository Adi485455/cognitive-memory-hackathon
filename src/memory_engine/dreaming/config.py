"""Configuration for Dream Engine execution."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DreamConfig:
    """Controls the limits of a dream execution."""

    max_memories_per_job: int = 100
    max_relationships_per_job: int = 50
    max_duplicate_candidates_per_job: int = 50
    max_conflict_candidates_per_job: int = 50
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for name in (
            "max_memories_per_job",
            "max_relationships_per_job",
            "max_duplicate_candidates_per_job",
            "max_conflict_candidates_per_job",
            "max_attempts",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1")