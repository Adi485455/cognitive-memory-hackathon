"""Dream Engine for memory consolidation."""

from memory_engine.dreaming.config import DreamConfig
from memory_engine.dreaming.engine import DreamEngine
from memory_engine.dreaming.models import (
    DreamJob,
    DreamJobStatus,
    DreamResult,
)
from memory_engine.dreaming.queue import DreamQueue
from memory_engine.dreaming.candidate_selector import (
    DreamCandidateSelector,
)
from memory_engine.dreaming.discovery import (
    ConflictCandidate,
    DiscoveryResult,
    DuplicateCandidate,
    RelationshipCandidate,
)
from memory_engine.dreaming.discovery_engine import MemoryDiscoveryEngine
from memory_engine.dreaming.consolidation import (
    ConsolidationDecisionEngine,
    ConsolidationResult,
)
from memory_engine.dreaming.orchestrator import (
    DreamExecution,
    DreamOrchestrator,
)
from memory_engine.dreaming.result_store import DreamResultStore

__all__ = [
    "DreamConfig",
    "DreamEngine",
    "DreamJob",
    "DreamJobStatus",
    "DreamQueue",
    "DreamResult",
    "DreamCandidateSelector",
    "ConflictCandidate",
    "DiscoveryResult",
    "DuplicateCandidate",
    "MemoryDiscoveryEngine",
    "RelationshipCandidate",
    "ConsolidationDecisionEngine",
    "ConsolidationResult",
    "DreamExecution",
    "DreamOrchestrator",
    "DreamResultStore",
]