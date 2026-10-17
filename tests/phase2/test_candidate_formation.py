import pytest

from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from memory_engine.kernel.models import MemoryType


def test_candidate_preserves_source_event_information():
    event = ExperienceIngestor().ingest(
        "I am building a cognitive memory system",
        source="conversation",
        scope="user-123",
        metadata={"topic": "memory"},
    )

    candidate = CandidateBuilder().build(event)

    assert candidate.content == event.content
    assert candidate.source_event_id == event.event_id
    assert candidate.source == event.source
    assert candidate.scope == event.scope
    assert candidate.metadata["topic"] == "memory"
    assert "source_occurred_at" in candidate.metadata


def test_candidate_uses_default_type_and_scores():
    event = ExperienceIngestor().ingest("Learning Python")

    candidate = CandidateBuilder().build(event)

    assert candidate.memory_type == MemoryType.SEMANTIC
    assert candidate.importance == 0.5
    assert candidate.confidence == 0.7


def test_candidate_accepts_custom_type_and_scores():
    event = ExperienceIngestor().ingest("A procedure for deployment")

    candidate = CandidateBuilder().build(
        event,
        memory_type=MemoryType.PROCEDURAL,
        importance=0.9,
        confidence=0.95,
    )

    assert candidate.memory_type == MemoryType.PROCEDURAL
    assert candidate.importance == 0.9
    assert candidate.confidence == 0.95


@pytest.mark.parametrize("importance", [-0.1, 1.1])
def test_candidate_rejects_invalid_importance(importance):
    event = ExperienceIngestor().ingest("A useful experience")

    with pytest.raises(ValueError):
        CandidateBuilder().build(event, importance=importance)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_candidate_rejects_invalid_confidence(confidence):
    event = ExperienceIngestor().ingest("A useful experience")

    with pytest.raises(ValueError):
        CandidateBuilder().build(event, confidence=confidence)