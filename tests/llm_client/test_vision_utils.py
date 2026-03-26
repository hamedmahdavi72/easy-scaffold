"""Tests for shared vision URL parsing."""

from __future__ import annotations

import base64

from easy_scaffold.llm_client.providers.vision_utils import parse_data_url_image


def test_parse_data_url_image_png() -> None:
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    b64 = base64.b64encode(raw).decode("ascii")
    url = f"data:image/png;base64,{b64}"
    got = parse_data_url_image(url)
    assert got is not None
    out_bytes, mime = got
    assert mime == "image/png"
    assert out_bytes == raw


def test_parse_data_url_image_rejects_non_data() -> None:
    assert parse_data_url_image("https://example.com/x.png") is None


def test_parse_data_url_image_rejects_non_base64() -> None:
    assert parse_data_url_image("data:text/plain,hello") is None
