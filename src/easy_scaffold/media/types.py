"""Stable JSON shape for image metadata stored on documents (not raw bytes)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    """Pointer to an object in blob storage."""

    key: str = Field(description="Object key within the configured bucket/prefix")
    bucket: Optional[str] = Field(
        default=None,
        description="Bucket name if different from default store bucket",
    )
    content_type: Optional[str] = Field(default=None, description="MIME type, e.g. image/png")
    etag: Optional[str] = Field(default=None, description="Store-specific version/etag if available")
