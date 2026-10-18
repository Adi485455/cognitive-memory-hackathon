"""Dream Engine orchestration foundation."""

from dataclasses import replace
from datetime import datetime, timezone

from memory_engine.dreaming.config import DreamConfig
from memory_engine.dreaming.models import (
    DreamJob,
    DreamJobStatus,
    DreamResult,
)


class DreamEngine:
    """Manage dream-job lifecycle and execution limits."""

    def __init__(self, config: DreamConfig | None = None) -> None:
        self.config = config or DreamConfig()

    def start_job(self, job: DreamJob) -> DreamJob:
        """Mark a pending job as running."""
        if job.attempt >= self.config.max_attempts:
            raise ValueError("Maximum job attempts reached")

        if job.status != DreamJobStatus.PENDING:
            raise ValueError("Only pending jobs can be started")

        return replace(
            job,
            status=DreamJobStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            attempt=job.attempt + 1,
            error=None,
    )

    def complete_job(
        self,
        job: DreamJob,
        result: DreamResult,
    ) -> DreamJob:
        """Mark a running job as completed."""
        if job.status != DreamJobStatus.RUNNING:
            raise ValueError("Only running jobs can be completed")

        if result.job_id != job.job_id:
            raise ValueError("Result job_id does not match job")

        return replace(
            job,
            status=DreamJobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
            error=None,
        )

    def fail_job(self, job: DreamJob, error: str) -> DreamJob:
        """Mark a running job as failed."""
        if job.status != DreamJobStatus.RUNNING:
            raise ValueError("Only running jobs can fail")

        if not error.strip():
            raise ValueError("error cannot be empty")

        return replace(
            job,
            status=DreamJobStatus.FAILED,
            error=error,
        )