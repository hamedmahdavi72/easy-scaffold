"""Helpers for OpenAI-style multimodal message parts (vision)."""

from __future__ import annotations

import base64
from typing import Optional, Tuple


def parse_data_url_image(url: str) -> Optional[Tuple[bytes, str]]:
    """Parse a ``data:`` URL with base64 payload; return ``(raw_bytes, mime_type)``."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    comma = url.find(",")
    if comma < 0:
        return None
    meta = url[5:comma]
    payload = url[comma + 1 :]
    if "base64" not in meta:
        return None
    mime = "application/octet-stream"
    if meta and meta.split(";")[0].strip():
        mime = meta.split(";")[0].strip()
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    return raw, mime
