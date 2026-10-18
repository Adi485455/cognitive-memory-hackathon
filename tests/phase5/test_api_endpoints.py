"""Tests for the versioned Cognitive Memory REST API."""

import pytest
from fastapi.testclient import TestClient

from memory_engine.api.app import app
from memory_engine.api.dependencies import (
    get_evidence_registry,
    get_memory_service,
)
from memory_engine.storage import (
    EvidenceRegistry,
    ObjectStore,
    PersistentMemoryService,
    SQLiteMemoryRepository,
)


@pytest.fixture
def client(tmp_path):
    """Create an API client backed by isolated temporary storage."""
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

    app.dependency_overrides[get_memory_service] = (
        lambda: memory_service
    )
    app.dependency_overrides[get_evidence_registry] = (
        lambda: evidence_registry
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def create_memory(client, content="A test canonical memory."):
    """Create a memory using the public API."""
    response = client.post(
        "/v1/memories",
        json={
            "content": content,
            "memory_type": "semantic",
            "source": "phase5-test",
            "scope": "test",
            "importance": 0.8,
            "confidence": 0.9,
            "metadata": {"test": True},
        },
    )

    assert response.status_code == 201, response.text
    return response.json()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "cognitive-memory-engine",
        "api_version": "v1",
    }


def test_create_and_get_memory(client):
    created = create_memory(client)

    assert created["content"] == "A test canonical memory."
    assert created["memory_type"] == "semantic"
    assert created["status"] == "active"
    assert created["version"] == 1
    assert len(created["evidence_ids"]) == 1

    response = client.get(
        f"/v1/memories/{created['memory_id']}"
    )

    assert response.status_code == 200
    assert response.json()["memory_id"] == created["memory_id"]
    assert response.json()["evidence_ids"] == created["evidence_ids"]


def test_list_memories(client):
    create_memory(client, "First test memory.")
    create_memory(client, "Second test memory.")

    response = client.get("/v1/memories")

    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 2
    assert len(body["items"]) == 2


def test_create_rejects_invalid_request(client):
    response = client.post(
        "/v1/memories",
        json={
            "content": "",
            "source": "phase5-test",
        },
    )

    assert response.status_code == 422


def test_get_memory_rejects_invalid_id(client):
    response = client.get("/v1/memories/not-a-uuid")

    assert response.status_code == 400


def test_get_memory_returns_404_for_missing_id(client):
    from uuid import uuid4

    response = client.get(f"/v1/memories/{uuid4()}")

    assert response.status_code == 404


def test_memory_history_contains_creation_event(client):
    created = create_memory(client)

    response = client.get(
        f"/v1/memories/{created['memory_id']}/history"
    )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["memory_id"] == created["memory_id"]
    assert body["count"] == 1
    assert len(body["items"]) == 1