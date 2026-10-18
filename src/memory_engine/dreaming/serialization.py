"""JSON serialization for Dream Engine execution reports."""

from __future__ import annotations

from typing import Any

from memory_engine.decision.jev import JEVAction, JEVDecision
from memory_engine.dreaming.consolidation import ConsolidationResult
from memory_engine.dreaming.discovery import (
    ConflictCandidate,
    DiscoveryResult,
    DuplicateCandidate,
    RelationshipCandidate,
)
from memory_engine.dreaming.models import DreamResult


def dream_execution_to_dict(
    result: DreamResult,
    discovery: DiscoveryResult,
    consolidation: ConsolidationResult,
) -> dict[str, Any]:
    """Serialize the report from one completed dreaming execution."""
    return {
        "result": {
            "job_id": result.job_id,
            "examined_memory_ids": list(result.examined_memory_ids),
            "discovered_relationships": result.discovered_relationships,
            "duplicate_candidates": result.duplicate_candidates,
            "conflict_candidates": result.conflict_candidates,
            "mutations_applied": result.mutations_applied,
            "warnings": list(result.warnings),
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
        },
        "discovery": {
            "duplicate_candidates": [
                {
                    "first_memory_id": item.first_memory_id,
                    "second_memory_id": item.second_memory_id,
                    "similarity": item.similarity,
                    "reason": item.reason,
                }
                for item in discovery.duplicate_candidates
            ],
            "relationship_candidates": [
                {
                    "first_memory_id": item.first_memory_id,
                    "second_memory_id": item.second_memory_id,
                    "relation_type": item.relation_type,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in discovery.relationship_candidates
            ],
            "conflict_candidates": [
                {
                    "first_memory_id": item.first_memory_id,
                    "second_memory_id": item.second_memory_id,
                    "reason": item.reason,
                    "confidence": item.confidence,
                }
                for item in discovery.conflict_candidates
            ],
            "examined_memory_ids": list(discovery.examined_memory_ids),
            "warnings": list(discovery.warnings),
        },
        "consolidation": {
            "decisions": [
                {
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "matched_memory_id": decision.matched_memory_id,
                    "target_memory_ids": list(decision.target_memory_ids),
                    "confidence": decision.confidence,
                    "requires_review": decision.requires_review,
                    "metadata": decision.metadata,
                }
                for decision in consolidation.decisions
            ],
        },
    }


def dream_execution_from_dict(
    payload: dict[str, Any],
) -> tuple[DreamResult, DiscoveryResult, ConsolidationResult]:
    """Reconstruct typed models from a persisted execution report."""
    result_data = payload["result"]
    discovery_data = payload["discovery"]
    consolidation_data = payload["consolidation"]

    result = DreamResult(
        job_id=result_data["job_id"],
        examined_memory_ids=tuple(result_data["examined_memory_ids"]),
        discovered_relationships=result_data["discovered_relationships"],
        duplicate_candidates=result_data["duplicate_candidates"],
        conflict_candidates=result_data["conflict_candidates"],
        mutations_applied=result_data["mutations_applied"],
        warnings=tuple(result_data["warnings"]),
        started_at=_parse_datetime(result_data["started_at"]),
        completed_at=_parse_datetime(result_data["completed_at"]),
    )

    discovery = DiscoveryResult(
        duplicate_candidates=tuple(
            DuplicateCandidate(**item)
            for item in discovery_data["duplicate_candidates"]
        ),
        relationship_candidates=tuple(
            RelationshipCandidate(**item)
            for item in discovery_data["relationship_candidates"]
        ),
        conflict_candidates=tuple(
            ConflictCandidate(**item)
            for item in discovery_data["conflict_candidates"]
        ),
        examined_memory_ids=tuple(discovery_data["examined_memory_ids"]),
        warnings=tuple(discovery_data["warnings"]),
    )

    consolidation = ConsolidationResult(
        decisions=tuple(
            JEVDecision(
                action=JEVAction(item["action"]),
                reason=item["reason"],
                matched_memory_id=item["matched_memory_id"],
                target_memory_ids=tuple(item["target_memory_ids"]),
                confidence=item["confidence"],
                requires_review=item["requires_review"],
                metadata=item["metadata"],
            )
            for item in consolidation_data["decisions"]
        ),
    )

    return result, discovery, consolidation


def _parse_datetime(value: str):
    """Parse an ISO-8601 timestamp."""
    from datetime import datetime

    return datetime.fromisoformat(value)