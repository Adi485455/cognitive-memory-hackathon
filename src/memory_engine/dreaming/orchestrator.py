"""Orchestrate a complete Dream Engine execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from memory_engine.dreaming.candidate_selector import (
    DreamCandidateSelector,
)
from memory_engine.dreaming.consolidation import (
    ConsolidationDecisionEngine,
    ConsolidationResult,
)
from memory_engine.dreaming.discovery import DiscoveryResult
from memory_engine.dreaming.discovery_engine import (
    MemoryDiscoveryEngine,
)
from memory_engine.dreaming.engine import DreamEngine
from memory_engine.dreaming.models import (
    DreamJob,
    DreamJobStatus,
    DreamResult,
)
from memory_engine.dreaming.queue import DreamQueue
from memory_engine.storage.persistent_memory_service import (
    PersistentMemoryService,
)
from memory_engine.dreaming.result_store import DreamResultStore
from memory_engine.dreaming.serialization import (
    dream_execution_from_dict,
    dream_execution_to_dict,
)


@dataclass(frozen=True, slots=True)
class DreamExecution:
    """Inspectable output of one complete dreaming pass."""

    job: DreamJob
    result: DreamResult
    discovery: DiscoveryResult
    consolidation: ConsolidationResult


class DreamOrchestrator:
    """Coordinate candidate selection, discovery, and decisions.

    The orchestrator proposes consolidation decisions but never applies
    them to canonical memories.
    """

    def __init__(
        self,
        *,
        queue: DreamQueue,
        memory_service: PersistentMemoryService,
        engine: DreamEngine | None = None,
        selector: DreamCandidateSelector | None = None,
        discovery_engine: MemoryDiscoveryEngine | None = None,
        consolidation_engine: ConsolidationDecisionEngine | None = None,
        result_store: DreamResultStore | None = None,
    ) -> None:
        self.queue = queue
        self.memory_service = memory_service
        self.engine = engine or DreamEngine()
        self.selector = selector or DreamCandidateSelector()
        self.discovery_engine = (
            discovery_engine or MemoryDiscoveryEngine()
        )
        self.consolidation_engine = (
            consolidation_engine or ConsolidationDecisionEngine()
        )
        self.result_store = result_store

    def run_next(self) -> DreamExecution | None:
        """Execute the oldest pending job.

        Returns None when the queue has no pending jobs.

        If execution fails, the job is requeued while attempts remain.
        Once the configured attempt limit is reached, it stays failed.
        The original exception is re-raised so callers can observe errors.
        """
        running = self.queue.claim_next(self.engine)

        if running is None:
            return None

        try:
            return self._execute(running)
        except Exception as exc:
            self._record_failure(running, exc)
            raise

    def _execute(self, running: DreamJob) -> DreamExecution:
        """Run the workflow for a job that has already been claimed."""
        memories = self.memory_service.list_memories()

        selected = self.selector.select(memories, running)

        discovery = self.discovery_engine.discover(selected)

        consolidation = self.consolidation_engine.evaluate(discovery)

        completed_at = datetime.now(timezone.utc)
        started_at = running.started_at or running.created_at

        result = DreamResult(
            job_id=running.job_id,
            examined_memory_ids=discovery.examined_memory_ids,
            discovered_relationships=len(
                discovery.relationship_candidates
            ),
            duplicate_candidates=len(
                discovery.duplicate_candidates
            ),
            conflict_candidates=len(
                discovery.conflict_candidates
            ),
            mutations_applied=0,
            warnings=discovery.warnings,
            started_at=started_at,
            completed_at=completed_at,
        )
        
        completed = self.engine.complete_job(running, result)

        execution = DreamExecution(
            job=completed,
            result=result,
            discovery=discovery,
            consolidation=consolidation,
        )

        # Persist the report before marking the job completed.
        if self.result_store is not None:
            payload = dream_execution_to_dict(
                execution.result,
                execution.discovery,
                execution.consolidation,
            )
            self.result_store.save(completed.job_id, payload)

        # Persist the completed job state.
        self.queue.save(completed)

        return execution

    def _record_failure(
        self,
        running: DreamJob,
        error: Exception,
    ) -> None:
        """Persist failure and requeue if attempts remain."""
        failed = self.engine.fail_job(
            running,
            str(error) or type(error).__name__,
        )

        max_attempts = self.engine.config.max_attempts

        if running.attempt < max_attempts:
            # Requeue without resetting the attempt count. The next claim
            # increments it, allowing the engine to enforce the limit.
            pending = replace(
                failed,
                status=DreamJobStatus.PENDING,
                started_at=None,
                completed_at=None,
            )
            self.queue.save(pending)
        else:
            self.queue.save(failed)

    def get_execution(self, job_id: str) -> DreamExecution:
        """Retrieve a previously persisted execution report."""
        if self.result_store is None:
            raise RuntimeError(
                "A DreamResultStore is required to retrieve execution reports"
            )

        payload = self.result_store.get(job_id)

        result, discovery, consolidation = dream_execution_from_dict(payload)

        job = self.queue.get(job_id)

        return DreamExecution(
            job=job,
            result=result,
            discovery=discovery,
            consolidation=consolidation,
        )