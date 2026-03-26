"""Unit tests for Gemini OpenAI-message conversion (no API calls)."""

from __future__ import annotations

import base64

from easy_scaffold.llm_client.providers.gemini import _openai_messages_to_contents


def test_gemini_user_multimodal_data_url() -> None:
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    b64 = base64.b64encode(raw).decode("ascii")
    url = f"data:image/png;base64,{b64}"
    system, contents = _openai_messages_to_contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
    )
    assert system is None
    assert len(contents) == 1
    parts = contents[0].parts
    assert len(parts) == 2
    assert parts[0].text == "describe"
    assert parts[1].inline_data is not None
    assert parts[1].inline_data.mime_type == "image/png"
    assert parts[1].inline_data.data == raw
