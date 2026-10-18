"""Versioned REST API for the Cognitive Memory Engine."""

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from memory_engine.api.dependencies import (
    get_evidence_registry,
    get_memory_orchestrator,
    get_memory_service,
)
from memory_engine.api.orchestrator import MemoryOrchestrator
from memory_engine.api.schemas import (
    ExperienceRequest,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
    OrchestrationResponse,
)
from memory_engine.kernel.ids import MemoryId
from memory_engine.kernel.models import Memory
from memory_engine.storage.canonical_repository import (
    MemoryVersionConflictError,
)
from memory_engine.storage.persistent_memory_service import (
    PersistentMemoryService,
)

app = FastAPI(
    title="Cognitive Memory Engine",
    description=(
        "A provenance-preserving cognitive memory API for LLM agents."
    ),
    version="1.0.0",
)


def serialize_memory(memory: Memory) -> MemoryResponse:
    """Convert a canonical memory to its public API representation."""

    return MemoryResponse(
        memory_id=str(memory.memory_id),
        content=memory.content,
        memory_type=memory.memory_type,
        status=memory.status,
        evidence_ids=[str(item) for item in memory.evidence_ids],
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        learned_at=memory.learned_at,
        version=memory.version,
        importance=memory.importance,
        confidence=memory.confidence,
        scope=memory.scope,
        metadata=memory.metadata,
    )


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is running."""

    return {
        "status": "ok",
        "service": "cognitive-memory-engine",
        "api_version": "v1",
    }


# ---------------------------------------------------------------------
# Create a canonical memory
# ---------------------------------------------------------------------

@app.post(
    "/v1/memories",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Memories"],
)
def create_memory(
    request: MemoryCreateRequest,
    memory_service: PersistentMemoryService = Depends(get_memory_service),
    evidence_registry: Any = Depends(get_evidence_registry),
) -> MemoryResponse:
    """Register evidence and create a canonical memory."""

    try:
        evidence = evidence_registry.register(
            request.content.encode("utf-8"),
            source=request.source,
            metadata={
                "scope": request.scope,
                **request.metadata,
            },
        )

        memory = Memory(
            memory_id=MemoryId.new(),
            content=request.content.strip(),
            memory_type=request.memory_type,
            evidence_ids=(evidence.evidence_id,),
            importance=request.importance,
            confidence=request.confidence,
            scope=request.scope,
            metadata=request.metadata,
        )

        created = memory_service.create(memory)

        return serialize_memory(created)

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# List canonical memories
# ---------------------------------------------------------------------

@app.get(
    "/v1/memories",
    response_model=MemoryListResponse,
    tags=["Memories"],
)
def list_memories(
    memory_service: PersistentMemoryService = Depends(get_memory_service),
) -> MemoryListResponse:
    """List canonical memories."""

    memories = [
        serialize_memory(memory)
        for memory in memory_service.list_memories()
    ]

    return MemoryListResponse(
        items=memories,
        count=len(memories),
    )


# ---------------------------------------------------------------------
# Retrieve one canonical memory
# ---------------------------------------------------------------------

@app.get(
    "/v1/memories/{memory_id}",
    response_model=MemoryResponse,
    tags=["Memories"],
)
def get_memory(
    memory_id: str,
    memory_service: PersistentMemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    """Retrieve one canonical memory."""

    try:
        parsed_id = MemoryId(value=memory_id)
        memory = memory_service.get(parsed_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID",
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        ) from exc

    return serialize_memory(memory)


# ---------------------------------------------------------------------
# Update a canonical memory
# ---------------------------------------------------------------------

@app.patch(
    "/v1/memories/{memory_id}",
    response_model=MemoryResponse,
    tags=["Memories"],
)
def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    memory_service: PersistentMemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    """Update a canonical memory using optimistic version control."""

    try:
        parsed_id = MemoryId(value=memory_id)

        updated = memory_service.update(
            parsed_id,
            expected_version=request.expected_version,
            content=request.content,
            importance=request.importance,
            confidence=request.confidence,
            metadata=request.metadata,
        )

        return serialize_memory(updated)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID or update values",
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        ) from exc

    except MemoryVersionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Retrieve a memory's event history
# ---------------------------------------------------------------------

@app.get(
    "/v1/memories/{memory_id}/history",
    tags=["Memories"],
)
def get_memory_history(
    memory_id: str,
    memory_service: PersistentMemoryService = Depends(get_memory_service),
) -> dict[str, Any]:
    """Return the durable domain events for one memory."""

    try:
        parsed_id = MemoryId(value=memory_id)

        # Verify that the requested memory exists.
        memory_service.get(parsed_id)

        events = memory_service.list_events_for_memory(parsed_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID",
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        ) from exc

    return {
        "memory_id": memory_id,
        "count": len(events),
        "items": [
            {
                "event_id": str(event.event_id),
                "event_type": event.event_type.value,
                "memory_version": event.memory_version,
                "occurred_at": event.occurred_at.isoformat(),
                "operation_id": event.operation_id,
                "metadata": event.metadata,
            }
            for event in events
        ],
    }


# ---------------------------------------------------------------------
# Process an experience through the Memory Orchestrator
# ---------------------------------------------------------------------

@app.post(
    "/v1/experiences",
    response_model=OrchestrationResponse,
    tags=["Memory Orchestration"],
)
def process_experience(
    request: ExperienceRequest,
    orchestrator: MemoryOrchestrator = Depends(get_memory_orchestrator),
) -> OrchestrationResponse:
    """Evaluate an experience and store it when safe."""

    try:
        result = orchestrator.process(
            request.content,
            source=request.source,
            scope=request.scope,
            memory_type=request.memory_type,
            importance=request.importance,
            confidence=request.confidence,
            metadata=request.metadata,
        )

        return OrchestrationResponse(
            event_id=result.event_id,
            action=result.action,
            reason=result.reason,
            confidence=result.confidence,
            requires_review=result.requires_review,
            memory=(
                serialize_memory(result.memory)
                if result.memory is not None
                else None
            ),
            evidence_id=result.evidence_id,
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc