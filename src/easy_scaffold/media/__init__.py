"""Blob storage and image references for workflow media."""

from .blob_store import AbstractBlobStore
from .types import ImageRef

__all__ = ["AbstractBlobStore", "ImageRef", "S3BlobStore"]

try:
    from .s3_blob_store import S3BlobStore
except ImportError:
    S3BlobStore = None  # type: ignore[misc, assignment]
