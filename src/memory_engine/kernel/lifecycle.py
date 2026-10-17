"""Memory lifecycle transition rules."""

from .models import MemoryStatus


_ALLOWED_TRANSITIONS: dict[MemoryStatus, frozenset[MemoryStatus]] = {
    MemoryStatus.ACTIVE: frozenset(
        {
            MemoryStatus.SUPERSEDED,
            MemoryStatus.ARCHIVED,
        }
    ),
    MemoryStatus.SUPERSEDED: frozenset(
        {
            MemoryStatus.ARCHIVED,
        }
    ),
    MemoryStatus.ARCHIVED: frozenset(),
}


def can_transition(
    current: MemoryStatus,
    target: MemoryStatus,
) -> bool:
    """Return whether a lifecycle transition is allowed."""
    return target in _ALLOWED_TRANSITIONS[current]


def transition(
    current: MemoryStatus,
    target: MemoryStatus,
) -> MemoryStatus:
    """Validate and return the target lifecycle status."""
    if current == target:
        raise ValueError(f"Memory is already in status {current.value}")

    if not can_transition(current, target):
        raise ValueError(
            f"Invalid memory lifecycle transition: "
            f"{current.value} -> {target.value}"
        )

    return target