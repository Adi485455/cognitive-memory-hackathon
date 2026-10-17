from memory_engine.decision.jev import JEV, JEVAction
from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from memory_engine.kernel.ids import EvidenceId, MemoryId
from memory_engine.kernel.manager import MemoryManager
from memory_engine.kernel.models import Memory, MemoryType


def build_candidate(content, *, confidence=0.7, scope="user"):
    event = ExperienceIngestor().ingest(content, scope=scope)

    return CandidateBuilder().build(
        event,
        memory_type=MemoryType.SEMANTIC,
        confidence=confidence,
    )


def create_memory(manager, content, *, scope="user"):
    memory = Memory(
        memory_id=MemoryId.new(),
        content=content,
        memory_type=MemoryType.SEMANTIC,
        evidence_ids=(EvidenceId.new(),),
        scope=scope,
    )
    return manager.create(memory)


def test_jev_stores_new_candidate():
    candidate = build_candidate("I am learning graph algorithms")

    decision = JEV().evaluate(candidate, ())

    assert decision.action == JEVAction.STORE
    assert decision.matched_memory_id is None


def test_jev_marks_low_confidence_candidate_uncertain():
    candidate = build_candidate(
        "Maybe this fact is correct",
        confidence=0.2,
    )

    decision = JEV().evaluate(candidate, ())

    assert decision.action == JEVAction.UNCERTAIN


def test_jev_ignores_duplicate_active_memory():
    manager = MemoryManager()
    memory = create_memory(
        manager,
        "I am learning graph algorithms",
    )

    candidate = build_candidate(
        "  I AM learning   graph algorithms  ",
        scope="user",
    )

    decision = JEV().evaluate(candidate, (memory,))

    assert decision.action == JEVAction.IGNORE
    assert decision.matched_memory_id == str(memory.memory_id)


def test_jev_does_not_ignore_memory_from_another_scope():
    manager = MemoryManager()
    memory = create_memory(
        manager,
        "I am learning graph algorithms",
        scope="user-A",
    )

    candidate = build_candidate(
        "I am learning graph algorithms",
        scope="user-B",
    )

    decision = JEV().evaluate(candidate, (memory,))

    assert decision.action == JEVAction.STORE

    from types import SimpleNamespace

import pytest

from memory_engine.decision import JEVAction, TypeSafeJEVAdapter
from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from types import SimpleNamespace


def _fake_response(
    action="STORE",
    score="Strong evidence",
    review_probability=0.1,
):
    return SimpleNamespace(
        model="jev-latest",
        choices={
            "action": SimpleNamespace(choice=action),
        },
        scores={
            "confidence": SimpleNamespace(score=score),
        },
        nouls={
            "needs_review": SimpleNamespace(noul=review_probability),
        },
    )


class _FakeTypeSafeClient:
    def __init__(self, response=None, error=None, **kwargs):
        self.response = response
        self.error = error
        self.received_arguments = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def system_one(self, **kwargs):
        self.received_arguments = kwargs

        if self.error:
            raise self.error

        return self.response


def _candidate():
    event = ExperienceIngestor().ingest(
        "I am building a cognitive memory system",
        scope="project",
    )
    return CandidateBuilder().build(event)


def test_typesafe_adapter_maps_store_decision(monkeypatch):
    response = _fake_response(action="STORE")

    monkeypatch.setattr(
        "memory_engine.decision.typesafe_adapter.TypeSafeClient",
        lambda **kwargs: _FakeTypeSafeClient(response=response, **kwargs),
    )

    decision = TypeSafeJEVAdapter().evaluate(_candidate(), ())

    assert decision.action == JEVAction.STORE
    assert decision.confidence == 0.9
    assert decision.requires_review is False
    assert decision.metadata["provider"] == "typesafe"


def test_typesafe_adapter_maps_supersede_and_requires_review(monkeypatch):
    response = _fake_response(action="SUPERSEDE")

    monkeypatch.setattr(
        "memory_engine.decision.typesafe_adapter.TypeSafeClient",
        lambda **kwargs: _FakeTypeSafeClient(response=response, **kwargs),
    )

    decision = TypeSafeJEVAdapter().evaluate(_candidate(), ())

    assert decision.action == JEVAction.SUPERSEDE
    assert decision.requires_review is True
    assert decision.metadata["target_selection_required"] is True


def test_typesafe_adapter_rejects_unknown_action(monkeypatch):
    response = _fake_response(action="DELETE_EVERYTHING")

    monkeypatch.setattr(
        "memory_engine.decision.typesafe_adapter.TypeSafeClient",
        lambda **kwargs: _FakeTypeSafeClient(response=response, **kwargs),
    )

    decision = TypeSafeJEVAdapter().evaluate(_candidate(), ())

    assert decision.action == JEVAction.UNCERTAIN
    assert decision.requires_review is True
    assert decision.confidence == 0.0


def test_typesafe_adapter_propagates_provider_errors(monkeypatch):
    def create_client(**kwargs):
        return _FakeTypeSafeClient(
            error=RuntimeError("simulated provider failure"),
            **kwargs,
        )

    monkeypatch.setattr(
        "memory_engine.decision.typesafe_adapter.TypeSafeClient",
        create_client,
    )

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        TypeSafeJEVAdapter().evaluate(_candidate(), ())