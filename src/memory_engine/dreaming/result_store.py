"""Persistent SQLite storage for Dream Engine execution results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any


class DreamResultStore:
    """Persist serialized Dream Engine execution reports."""

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
                CREATE TABLE IF NOT EXISTS dream_results (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, job_id: str, payload: dict[str, Any]) -> None:
        """Save a result once; reject conflicting duplicate writes."""
        if not job_id.strip():
            raise ValueError("job_id cannot be empty")

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                existing = connection.execute(
                    """
                    SELECT payload_json
                    FROM dream_results
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()

                if existing is not None:
                    if existing["payload_json"] != serialized:
                        raise ValueError(
                            f"A different result already exists for job: "
                            f"{job_id}"
                        )

                    connection.commit()
                    return

                connection.execute(
                    """
                    INSERT INTO dream_results (job_id, payload_json)
                    VALUES (?, ?)
                    """,
                    (job_id, serialized),
                )
                connection.commit()

            except Exception:
                connection.rollback()
                raise

    def get(self, job_id: str) -> dict[str, Any]:
        """Retrieve a stored result payload."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json
                FROM dream_results
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Dream result not found: {job_id}")

        return json.loads(row["payload_json"])

    def exists(self, job_id: str) -> bool:
        """Check whether a result has been stored."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM dream_results
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

        return row is not None