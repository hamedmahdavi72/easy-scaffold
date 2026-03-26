"""Resolve ``ImageRef`` from context into OpenAI-style multimodal message parts."""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

from .blob_store import AbstractBlobStore
from .types import ImageRef

logger = logging.getLogger(__name__)


async def append_image_to_message(
    message: Dict[str, Any],
    blob_store: AbstractBlobStore,
    ref: Any,
) -> None:
    """Mutate ``message`` in place: add an ``image_url`` part (data URL) after text."""
    ir = ImageRef.model_validate(ref)
    data = await blob_store.get_bytes(ir.key, bucket=ir.bucket)
    mime = ir.content_type or "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    url = f"data:{mime};base64,{b64}"
    img_part: Dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
    c = message.get("content", "")
    if isinstance(c, str):
        parts: List[Dict[str, Any]] = [{"type": "text", "text": c}, img_part]
    elif isinstance(c, list):
        parts = list(c) + [img_part]
    else:
        parts = [{"type": "text", "text": str(c)}, img_part]
    message["content"] = parts


async def apply_media_attachments(
    messages: List[Dict[str, Any]],
    attachments: List[Any],
    context: Dict[str, Any],
    blob_store: AbstractBlobStore,
    get_from_nested_dict: Any,
) -> List[Dict[str, Any]]:
    """Return a shallow copy of ``messages`` with images merged into target indices."""
    out = [dict(m) for m in messages]
    for att in attachments:
        source = att.source
        idx = att.message_index
        if idx < 0 or idx >= len(out):
            raise ValueError(f"media_attachment message_index {idx} out of range")
        ref = get_from_nested_dict(context, source)
        if ref is None:
            raise ValueError(f"media_attachment source {source!r} resolved to None")
        msg = dict(out[idx])
        await append_image_to_message(msg, blob_store, ref)
        out[idx] = msg
    return out
