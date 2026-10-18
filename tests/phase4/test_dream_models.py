import pytest
from dataclasses import replace
from datetime import datetime, timezone

from memory_engine.dreaming.models import (
    DreamJob,
    DreamJobStatus,
    DreamResult,
)


def test_dream_job_defaults_to_pending():
    job = DreamJob()

    assert job.status == DreamJobStatus.PENDING
    assert job.attempt == 0
    assert job.started_at is None
    assert job.completed_at is None


def test_dream_job_rejects_negative_attempt():
    with pytest.raises(ValueError, match="attempt"):
        DreamJob(attempt=-1)


def test_running_job_requires_started_at():
    with pytest.raises(ValueError, match="started_at"):
        DreamJob(status=DreamJobStatus.RUNNING)


def test_completed_job_requires_completed_at():
    with pytest.raises(ValueError, match="completed_at"):
        DreamJob(status=DreamJobStatus.COMPLETED)


def test_dream_result_rejects_negative_mutation_count():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="mutations_applied"):
        DreamResult(
            job_id="job-1",
            mutations_applied=-1,
            started_at=now,
            completed_at=now,
        )


def test_dream_result_rejects_completion_before_start():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="precede"):
        DreamResult(
            job_id="job-1",
            started_at=now,
            completed_at=datetime(
                2000, 1, 1, tzinfo=timezone.utc
            ),
        )