"""S3-backed blob storage using boto3 (blocking calls run in a thread)."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .blob_store import AbstractBlobStore
from .types import ImageRef

logger = logging.getLogger(__name__)


class S3BlobStore(AbstractBlobStore):
    """Async-friendly wrapper around boto3 S3."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        region_name: Optional[str] = None,
    ) -> None:
        self._bucket = bucket
        p = prefix.strip().strip("/")
        self._prefix = f"{p}/" if p else ""
        kwargs = {}
        if region_name:
            kwargs["region_name"] = region_name
        self._client = boto3.client("s3", **kwargs)

    def _full_key(self, key: str) -> str:
        k = key.lstrip("/")
        return f"{self._prefix}{k}" if self._prefix else k

    async def get_bytes(self, key: str, *, bucket: str | None = None) -> bytes:
        """``key`` is the full object key (same as ``ImageRef.key`` from ``put_bytes``)."""
        b = bucket or self._bucket

        def _get() -> bytes:
            resp = self._client.get_object(Bucket=b, Key=key)
            return resp["Body"].read()

        try:
            return await asyncio.to_thread(_get)
        except ClientError:
            logger.exception("S3 get_object failed: bucket=%s key=%s", b, key)
            raise

    async def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str,
        *,
        bucket: str | None = None,
    ) -> ImageRef:
        b = bucket or self._bucket
        fk = self._full_key(key)

        def _put() -> str:
            resp = self._client.put_object(
                Bucket=b,
                Key=fk,
                Body=data,
                ContentType=content_type,
            )
            return str(resp.get("ETag", "") or "").strip('"')

        try:
            etag = await asyncio.to_thread(_put)
        except ClientError:
            logger.exception("S3 put_object failed: bucket=%s key=%s", b, fk)
            raise
        return ImageRef(key=fk, bucket=b, content_type=content_type, etag=etag or None)
