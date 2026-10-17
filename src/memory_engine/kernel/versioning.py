"""Memory version validation utilities."""


class VersionConflictError(RuntimeError):
    """Raised when a mutation targets a stale memory version."""


def validate_expected_version(
    current_version: int,
    expected_version: int,
) -> None:
    """Reject a mutation if its expected version is stale."""
    if current_version < 1:
        raise ValueError("Current version must be at least 1")

    if expected_version < 1:
        raise ValueError("Expected version must be at least 1")

    if current_version != expected_version:
        raise VersionConflictError(
            f"Expected memory version {expected_version}, "
            f"but current version is {current_version}"
        )


def next_version(current_version: int) -> int:
    """Return the next sequential memory version."""
    if current_version < 1:
        raise ValueError("Current version must be at least 1")

    return current_version + 1