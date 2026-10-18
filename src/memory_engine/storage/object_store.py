"""Content-addressed storage for immutable evidence objects."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class ObjectStore:
    """Store and retrieve raw evidence using SHA-256 content hashes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def content_hash(content: bytes) -> str:
        """Return the SHA-256 hash of the supplied bytes."""
        return hashlib.sha256(content).hexdigest()

    def _path_for(self, content_hash: str) -> Path:
        """Resolve a valid SHA-256 hash to its object path."""
        if (
            len(content_hash) != 64
            or any(c not in "0123456789abcdef" for c in content_hash)
        ):
            raise ValueError("Invalid SHA-256 content hash")

        return self.root / content_hash[:2] / content_hash[2:]

    def put(self, content: bytes) -> str:
        """Persist content and return its SHA-256 hash."""
        if not isinstance(content, bytes):
            raise TypeError("Object content must be bytes")

        digest = self.content_hash(content)
        destination = self._path_for(digest)

        if destination.exists():
            return digest

        destination.parent.mkdir(parents=True, exist_ok=True)

        fd, temporary_name = tempfile.mkstemp(
            prefix=".object-",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            # Atomic creation without overwriting an existing object.
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)

        return digest

    def get(self, content_hash: str) -> bytes:
        """Retrieve an object and verify its content hash."""
        path = self._path_for(content_hash)

        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(
                f"Object not found: {content_hash}"
            ) from exc

        actual_hash = self.content_hash(content)

        if actual_hash != content_hash:
            raise ValueError("Stored object failed content hash verification")

        return content

    def exists(self, content_hash: str) -> bool:
        """Return whether the object exists."""
        return self._path_for(content_hash).is_file()