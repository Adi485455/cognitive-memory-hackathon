import pytest

from memory_engine.kernel.lifecycle import (
    can_transition,
    transition,
)
from memory_engine.kernel.models import MemoryStatus
from memory_engine.kernel.versioning import (
    VersionConflictError,
    next_version,
    validate_expected_version,
)


def test_active_memory_can_be_superseded():
    assert can_transition(
        MemoryStatus.ACTIVE,
        MemoryStatus.SUPERSEDED,
    )


def test_archived_memory_cannot_be_reactivated():
    assert not can_transition(
        MemoryStatus.ARCHIVED,
        MemoryStatus.ACTIVE,
    )


def test_invalid_transition_raises_error():
    with pytest.raises(ValueError, match="Invalid"):
        transition(
            MemoryStatus.ARCHIVED,
            MemoryStatus.ACTIVE,
        )


def test_version_checks_reject_stale_updates():
    validate_expected_version(3, 3)

    with pytest.raises(VersionConflictError):
        validate_expected_version(4, 3)


def test_next_version_increments():
    assert next_version(1) == 2
    assert next_version(4) == 5