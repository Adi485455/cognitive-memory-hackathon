"""Request and response schemas for the versioned Memory API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_engine.kernel.models import MemoryStatus, MemoryType


class MemoryCreateRequest(BaseModel):
    """Request to create a canonical memory with supporting evidence."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=100_000)
    memory_type: MemoryType = MemoryType.SEMANTIC
    source: str = Field(min_length=1, max_length=500)
    scope: str = Field(default="user", min_length=1, max_length=200)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Public representation of a canonical memory."""

    memory_id: str
    content: str
    memory_type: MemoryType
    status: MemoryStatus
    evidence_ids: list[str]
    created_at: datetime
    updated_at: datetime
    learned_at: datetime
    version: int
    importance: float
    confidence: float
    scope: str
    metadata: dict[str, Any]


class MemoryListResponse(BaseModel):
    """A collection of canonical memories."""

    items: list[MemoryResponse]
    count: int

class MemoryUpdateRequest(BaseModel):
    """Request to update an existing canonical memory."""

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    content: str | None = Field(default=None, min_length=1, max_length=100_000)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = None

class ExperienceRequest(BaseModel):
    """Incoming experience to be evaluated for memory storage."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=100_000)
    source: str = Field(default="conversation", min_length=1, max_length=500)
    scope: str = Field(default="user", min_length=1, max_length=200)
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrationResponse(BaseModel):
    """Outcome of evaluating an incoming experience."""

    event_id: str
    action: str
    reason: str
    confidence: float
    requires_review: bool
    memory: MemoryResponse | None = None
    evidence_id: str | None = None