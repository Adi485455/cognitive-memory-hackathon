"""Persistent SQLite queue for Dream Engine jobs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock

from memory_engine.dreaming.models import (
    DreamJob,
    DreamJobStatus,
)


class DreamQueue:
    """Persist and coordinate Dream Engine jobs."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._lock = RLock()

        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dream_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    trigger_key TEXT UNIQUE,
                    record_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dream_jobs_pending
                ON dream_jobs(status, created_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _trigger_key(job: DreamJob) -> str | None:
        """Return a stable key for jobs with trigger events."""
        if not job.trigger_event_ids:
            return None

        normalized = sorted(set(job.trigger_event_ids))
        return json.dumps(normalized, separators=(",", ":"))

    @staticmethod
    def _serialize(job: DreamJob) -> str:
        return json.dumps(
            {
                "job_id": job.job_id,
                "target_memory_ids": list(job.target_memory_ids),
                "trigger_event_ids": list(job.trigger_event_ids),
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": (
                    job.started_at.isoformat()
                    if job.started_at
                    else None
                ),
                "completed_at": (
                    job.completed_at.isoformat()
                    if job.completed_at
                    else None
                ),
                "attempt": job.attempt,
                "error": job.error,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _deserialize(record_json: str) -> DreamJob:
        data = json.loads(record_json)

        return DreamJob(
            job_id=data["job_id"],
            target_memory_ids=tuple(data["target_memory_ids"]),
            trigger_event_ids=tuple(data["trigger_event_ids"]),
            status=DreamJobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
            attempt=data["attempt"],
            error=data["error"],
        )

    def enqueue(self, job: DreamJob) -> DreamJob:
        """Persist a pending job, deduplicating its trigger set."""
        if job.status != DreamJobStatus.PENDING:
            raise ValueError("Only pending jobs can be enqueued")

        trigger_key = self._trigger_key(job)

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                if trigger_key is not None:
                    existing = connection.execute(
                        """
                        SELECT record_json
                        FROM dream_jobs
                        WHERE trigger_key = ?
                        """,
                        (trigger_key,),
                    ).fetchone()

                    if existing is not None:
                        connection.commit()
                        return self._deserialize(
                            existing["record_json"]
                        )

                connection.execute(
                    """
                    INSERT INTO dream_jobs (
                        job_id,
                        status,
                        created_at,
                        attempt,
                        trigger_key,
                        record_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.status.value,
                        job.created_at.isoformat(),
                        job.attempt,
                        trigger_key,
                        self._serialize(job),
                    ),
                )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return job

    def get(self, job_id: str) -> DreamJob:
        """Retrieve a job by identifier."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_json
                FROM dream_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Dream job not found: {job_id}")

        return self._deserialize(row["record_json"])

    def list_jobs(
        self,
        status: DreamJobStatus | None = None,
    ) -> tuple[DreamJob, ...]:
        """List jobs in creation order, optionally filtered by status."""
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT record_json
                    FROM dream_jobs
                    ORDER BY created_at, job_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT record_json
                    FROM dream_jobs
                    WHERE status = ?
                    ORDER BY created_at, job_id
                    """,
                    (status.value,),
                ).fetchall()

        return tuple(
            self._deserialize(row["record_json"])
            for row in rows
        )

    def claim_next(self, engine) -> DreamJob | None:
        """Atomically claim the oldest pending job."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                row = connection.execute(
                    """
                    SELECT job_id, record_json
                    FROM dream_jobs
                    WHERE status = ?
                    ORDER BY created_at, job_id
                    LIMIT 1
                    """,
                    (DreamJobStatus.PENDING.value,),
                ).fetchone()

                if row is None:
                    connection.commit()
                    return None

                pending = self._deserialize(row["record_json"])
                running = engine.start_job(pending)

                connection.execute(
                    """
                    UPDATE dream_jobs
                    SET status = ?, attempt = ?, record_json = ?
                    WHERE job_id = ? AND status = ?
                    """,
                    (
                        running.status.value,
                        running.attempt,
                        self._serialize(running),
                        running.job_id,
                        DreamJobStatus.PENDING.value,
                    ),
                )

                if connection.total_changes == 0:
                    raise RuntimeError("Failed to claim dream job")

                connection.commit()
                return running

            except Exception:
                connection.rollback()
                raise

    def save(self, job: DreamJob) -> DreamJob:
        """Persist a job's latest state."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                cursor = connection.execute(
                    """
                    UPDATE dream_jobs
                    SET status = ?, attempt = ?, record_json = ?
                    WHERE job_id = ?
                    """,
                    (
                        job.status.value,
                        job.attempt,
                        self._serialize(job),
                        job.job_id,
                    ),
                )

                if cursor.rowcount != 1:
                    raise KeyError(
                        f"Dream job not found: {job.job_id}"
                    )

                connection.commit()

            except Exception:
                connection.rollback()
                raise

        return job