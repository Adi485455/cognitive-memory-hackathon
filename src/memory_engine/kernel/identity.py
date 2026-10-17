"""Typed identity creation and validation helpers."""

from typing import TypeVar

from .ids import (
    DecisionId,
    EntityId,
    EventId,
    EvidenceId,
    MemoryId,
    RelationshipId,
    StableId,
)


TStableId = TypeVar("TStableId", bound=StableId)


def new_id(id_type: type[TStableId]) -> TStableId:
    """Create a new identifier of the requested domain type."""
    return id_type.new()


def require_id_type(value: object, expected_type: type[TStableId]) -> TStableId:
    """Validate that a value has the expected domain ID type."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, "
            f"received {type(value).__name__}"
        )

    return value


__all__ = [
    "DecisionId",
    "EntityId",
    "EventId",
    "EvidenceId",
    "MemoryId",
    "RelationshipId",
    "new_id",
    "require_id_type",
]