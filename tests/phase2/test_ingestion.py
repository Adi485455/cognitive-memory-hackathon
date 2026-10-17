from datetime import datetime, timezone

import pytest

from memory_engine.formation.ingestion import ExperienceIngestor


def test_ingestion_normalizes_whitespace():
    event = ExperienceIngestor().ingest(
        "  I am   learning   machine learning.  "
    )

    assert event.content == "I am learning machine learning."


def test_ingestion_generates_unique_event_ids():
    ingestor = ExperienceIngestor()

    first = ingestor.ingest("First experience")
    second = ingestor.ingest("Second experience")

    assert first.event_id != second.event_id


def test_ingestion_preserves_source_scope_and_metadata():
    event = ExperienceIngestor().ingest(
        "Building a memory system",
        source="conversation",
        scope="user-123",
        metadata={"topic": "AI"},
    )

    assert event.source == "conversation"
    assert event.scope == "user-123"
    assert event.metadata["topic"] == "AI"


def test_ingestion_uses_timezone_aware_timestamp():
    event = ExperienceIngestor().ingest("A useful experience")

    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() is not None


def test_ingestion_accepts_explicit_timestamp():
    timestamp = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)

    event = ExperienceIngestor().ingest(
        "A dated experience",
        occurred_at=timestamp,
    )

    assert event.occurred_at == timestamp


def test_ingestion_rejects_empty_content():
    with pytest.raises(ValueError):
        ExperienceIngestor().ingest("   ")


def test_ingestion_rejects_non_string_content():
    with pytest.raises(TypeError):
        ExperienceIngestor().ingest(None)