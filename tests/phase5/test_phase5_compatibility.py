"""Compatibility tests for Phase 5 and the existing memory architecture."""

from fastapi.testclient import TestClient

from memory_engine.api.app import app
from memory_engine.api.dependencies import (
    get_evidence_registry,
    get_memory_orchestrator,
    get_memory_service,
)
from memory_engine.api.orchestrator import MemoryOrchestrator
from memory_engine.decision.jev import JEV, JEVAction
from memory_engine.formation.candidate import CandidateBuilder
from memory_engine.formation.ingestion import ExperienceIngestor
from memory_engine.kernel.models import MemoryType
from memory_engine.storage import (
    EvidenceRegistry,
    ObjectStore,
    PersistentMemoryService,
    SQLiteMemoryRepository,
)


def build_test_dependencies(tmp_path):
    """Construct isolated dependencies for integration tests."""

    data_dir = tmp_path / "memory_data"
    data_dir.mkdir()

    repository = SQLiteMemoryRepository(
        data_dir / "memories.sqlite3"
    )
    memory_service = PersistentMemoryService(repository)

    evidence_registry = EvidenceRegistry(
        data_dir / "evidence",
        ObjectStore(data_dir / "objects"),
    )

    orchestrator = MemoryOrchestrator(
        memory_service=memory_service,
        evidence_registry=evidence_registry,
    )

    return memory_service, evidence_registry, orchestrator


def test_phase5_orchestrator_uses_existing_formation_and_decision_layers(
    tmp_path,
):
    memory_service, evidence_registry, orchestrator = (
        build_test_dependencies(tmp_path)
    )

    assert isinstance(orchestrator.ingestor, ExperienceIngestor)
    assert isinstance(orchestrator.candidate_builder, CandidateBuilder)
    assert isinstance(orchestrator.jev, JEV)

    result = orchestrator.process(
        "The user prefers detailed technical explanations.",
        source="compatibility-test",
        scope="compatibility-user",
        memory_type=MemoryType.PREFERENCE,
        confidence=0.9,
    )

    assert result.action == JEVAction.STORE.value
    assert result.memory is not None
    assert result.memory.memory_type == MemoryType.PREFERENCE

    assert len(memory_service.list_memories()) == 1
    assert result.evidence_id is not None


def test_phase5_storage_preserves_canonical_memory_and_event_history(
    tmp_path,
):
    memory_service, _, orchestrator = build_test_dependencies(
        tmp_path
    )

    result = orchestrator.process(
        "The user prefers concise explanations.",
        source="compatibility-test",
        scope="compatibility-user",
        confidence=0.9,
    )

    assert result.memory is not None

    memory_id = result.memory.memory_id

    restored_memory = memory_service.get(memory_id)
    events = memory_service.list_events_for_memory(memory_id)

    assert restored_memory.memory_id == memory_id
    assert restored_memory.content == (
        "The user prefers concise explanations."
    )
    assert restored_memory.evidence_ids == result.memory.evidence_ids

    assert len(events) == 1
    assert events[0].memory_version == restored_memory.version


def test_phase5_api_works_with_existing_memory_services(tmp_path):
    memory_service, evidence_registry, orchestrator = (
        build_test_dependencies(tmp_path)
    )

    app.dependency_overrides[get_memory_service] = (
        lambda: memory_service
    )
    app.dependency_overrides[get_evidence_registry] = (
        lambda: evidence_registry
    )
    app.dependency_overrides[get_memory_orchestrator] = (
        lambda: orchestrator
    )

    try:
        with TestClient(app) as client:
            health_response = client.get("/health")

            assert health_response.status_code == 200
            assert health_response.json()["api_version"] == "v1"

            create_response = client.post(
                "/v1/experiences",
                json={
                    "content": "The user prefers structured code examples.",
                    "source": "compatibility-test",
                    "scope": "compatibility-user",
                    "memory_type": "preference",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "metadata": {"test": "phase5-compatibility"},
                },
            )

            assert create_response.status_code == 200, (
                create_response.text
            )

            body = create_response.json()

            assert body["action"] == "store"
            assert body["memory"] is not None
            assert body["evidence_id"] is not None

            memory_id = body["memory"]["memory_id"]

            get_response = client.get(
                f"/v1/memories/{memory_id}"
            )

            assert get_response.status_code == 200
            assert get_response.json()["memory_id"] == memory_id

            history_response = client.get(
                f"/v1/memories/{memory_id}/history"
            )

            assert history_response.status_code == 200
            assert history_response.json()["count"] == 1

    finally:
        app.dependency_overrides.clear()


def test_low_confidence_decision_does_not_mutate_canonical_state(
    tmp_path,
):
    memory_service, _, orchestrator = build_test_dependencies(
        tmp_path
    )

    result = orchestrator.process(
        "The user might prefer a different programming language.",
        source="compatibility-test",
        scope="compatibility-user",
        confidence=0.2,
    )

    assert result.action == JEVAction.UNCERTAIN.value
    assert result.requires_review is True
    assert result.memory is None

    assert memory_service.list_memories() == ()
    assert memory_service.list_events() == ()


def test_existing_dream_engine_package_remains_importable():
    """Verify that Phase 5 can coexist with the Phase 4 Dream Engine."""

    from memory_engine.dreaming.config import DreamConfig
    from memory_engine.dreaming.engine import DreamEngine
    from memory_engine.dreaming.models import DreamResult

    assert DreamConfig is not None
    assert DreamEngine is not None
    assert DreamResult is not None