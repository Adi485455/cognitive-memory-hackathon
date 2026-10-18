"""Select canonical memories for Dream Engine processing."""

from __future__ import annotations

from memory_engine.dreaming.config import DreamConfig
from memory_engine.dreaming.models import DreamJob
from memory_engine.kernel.models import Memory, MemoryStatus


class DreamCandidateSelector:
    """Select eligible memories deterministically for a dream job."""

    def __init__(self, config: DreamConfig | None = None) -> None:
        self.config = config or DreamConfig()

    def select(
        self,
        memories: tuple[Memory, ...] | list[Memory],
        job: DreamJob,
    ) -> tuple[Memory, ...]:
        """Return eligible memories without modifying canonical state.

        Selection rules:
        1. Only active memories are eligible.
        2. Explicit job targets restrict the candidate set.
        3. Candidates are ordered deterministically.
        4. The configured per-job limit is always respected.
        """
        target_ids = set(job.target_memory_ids)

        eligible = [
            memory
            for memory in memories
            if memory.status == MemoryStatus.ACTIVE
            and (
                not target_ids
                or str(memory.memory_id) in target_ids
            )
        ]

        # Stable ordering makes runs reproducible.
        eligible.sort(
            key=lambda memory: (
                -memory.importance,
                -memory.confidence,
                memory.created_at,
                str(memory.memory_id),
            )
        )

        return tuple(
            eligible[: self.config.max_memories_per_job]
        )