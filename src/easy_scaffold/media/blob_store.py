"""Abstract async blob storage (S3, local, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import ImageRef


class AbstractBlobStore(ABC):
    """Load and store binary objects by key."""

    @abstractmethod
    async def get_bytes(self, key: str, *, bucket: str | None = None) -> bytes:
        """Return object bytes. ``bucket`` overrides default when store supports it."""

    @abstractmethod
    async def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str,
        *,
        bucket: str | None = None,
    ) -> ImageRef:
        """Persist bytes and return a reference suitable for Mongo/DocumentDB."""
