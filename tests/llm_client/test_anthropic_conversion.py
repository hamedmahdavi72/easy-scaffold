"""Unit tests for Anthropic provider message/tool conversion (no API calls)."""

from __future__ import annotations

from anthropic.types import Message
from pydantic import BaseModel

from easy_scaffold.llm_client.providers.anthropic import (
    anthropic_message_to_chat_result,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)


def test_openai_tools_to_anthropic_maps_function_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
            },
        }
    ]
    out = openai_tools_to_anthropic(tools)
    assert out is not None
    assert len(out) == 1
    assert out[0]["name"] == "add"
    assert out[0]["description"] == "Add two numbers"
    assert out[0]["input_schema"]["type"] == "object"


def test_openai_messages_system_user() -> None:
    system, msgs = openai_messages_to_anthropic(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert system == "You are helpful."
    assert msgs == [{"role": "user", "content": "Hi"}]


def test_openai_messages_merges_consecutive_user() -> None:
    _, msgs = openai_messages_to_anthropic(
        [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "a" in msgs[0]["content"] and "b" in msgs[0]["content"]


def test_openai_messages_tool_round_trip_shape() -> None:
    _, msgs = openai_messages_to_anthropic(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 1}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "add",
                "content": '{"result": 2}',
            },
        ]
    )
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    blocks = msgs[1]["content"]
    assert isinstance(blocks, list)
    tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["id"] == "call_abc"
    assert tool_blocks[0]["name"] == "add"
    assert tool_blocks[0]["input"] == {"a": 1}
    assert msgs[2]["role"] == "user"
    tr = msgs[2]["content"]
    assert tr[0]["type"] == "tool_result"
    assert tr[0]["tool_use_id"] == "call_abc"


def test_anthropic_message_to_chat_result_tool_use_openai_shape() -> None:
    msg = Message.model_validate(
        {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "add",
                    "input": {"a": 3},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    result = anthropic_message_to_chat_result(msg)
    assert len(result.choices) == 1
    tc = result.choices[0].message.tool_calls
    assert tc is not None and len(tc) == 1
    assert tc[0].id == "toolu_01"
    assert tc[0].function.name == "add"
    assert '"a": 3' in tc[0].function.arguments


def test_anthropic_message_to_chat_result_structured_parsed() -> None:
    class Out(BaseModel):
        answer: int

    msg = Message.model_validate(
        {
            "id": "m2",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "text", "text": '{"answer": 42}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    result = anthropic_message_to_chat_result(msg, response_model_cls=Out)
    parsed = result.choices[0].message.parsed
    assert isinstance(parsed, Out)
    assert parsed.answer == 42


def test_openai_messages_user_with_image_data_url() -> None:
    import base64

    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    b64 = base64.b64encode(raw).decode("ascii")
    url = f"data:image/png;base64,{b64}"
    _, msgs = openai_messages_to_anthropic(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ]
    )
    assert len(msgs) == 1
    c = msgs[0]["content"]
    assert isinstance(c, list)
    assert c[0]["type"] == "text"
    assert c[1]["type"] == "image"
    assert c[1]["source"]["type"] == "base64"
    assert c[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(c[1]["source"]["data"]) == raw


def test_openai_messages_user_image_http_url() -> None:
    _, msgs = openai_messages_to_anthropic(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    }
                ],
            }
        ]
    )
    c = msgs[0]["content"]
    assert isinstance(c, list)
    assert c[0]["type"] == "image"
    assert c[0]["source"]["type"] == "url"
