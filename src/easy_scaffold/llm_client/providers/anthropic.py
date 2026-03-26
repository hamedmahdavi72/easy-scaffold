"""Anthropic Messages API (Claude) via official async SDK."""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from anthropic import APIStatusError, AsyncAnthropic, RateLimitError
from anthropic import APIError as AnthropicAPIError
from pydantic import BaseModel

from ...configs.pydantic_models import ModelProfile
from .errors import ProviderAPIError, ProviderContentPolicyError, ProviderRateLimitError
from .vision_utils import parse_data_url_image
from .response import (
    ChatCompletionResult,
    _ChatChoice,
    _ChatMessage,
    _UsageStats,
)

logger = logging.getLogger(__name__)

_SKIP_KEYS = frozenset(
    {
        "allowed_openai_params",
        "max_output_tokens",
        "thinking",
        "response_mime_type",
        "reasoning_effort",
    }
)

_ALLOWED_MESSAGE_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "metadata",
        "tool_choice",
    }
)


def _strip_model_id(model: str) -> str:
    if model.startswith("anthropic/"):
        return model[len("anthropic/") :]
    return model


def _tool_call_id(raw: Any) -> str:
    if raw:
        return str(raw)
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _text_from_openai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content)


def _openai_user_content_to_anthropic(content: Any) -> Any:
    """Map OpenAI user content (str or multimodal list) to Anthropic user content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    blocks: List[Dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            blocks.append({"type": "text", "text": str(block.get("text") or "")})
        elif bt == "image_url":
            iu = block.get("image_url") or {}
            url = iu.get("url") if isinstance(iu, dict) else None
            if not url:
                continue
            url_s = str(url)
            if url_s.startswith(("http://", "https://")):
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": url_s}}
                )
                continue
            parsed = parse_data_url_image(url_s)
            if not parsed:
                continue
            raw_b, mime = parsed
            b64 = base64.b64encode(raw_b).decode("ascii")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                }
            )
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return blocks[0].get("text") or ""
    return blocks


def openai_tools_to_anthropic(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        entry: Dict[str, Any] = {
            "name": name,
            "input_schema": params,
        }
        desc = fn.get("description")
        if desc:
            entry["description"] = desc
        out.append(entry)
    return out or None


def openai_messages_to_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Map OpenAI-style chat messages to Anthropic ``system`` + ``messages``."""
    system_chunks: List[str] = []
    anth_msgs: List[Dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                system_chunks.append(c)
            i += 1
            continue
        if role == "user":
            ucontent = _openai_user_content_to_anthropic(m.get("content"))
            anth_msgs.append({"role": "user", "content": ucontent})
            i += 1
            continue
        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _text_from_openai_content(m.get("content"))
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    args_raw = fn.get("arguments") or "{}"
                    tid = tc.get("id")
                else:
                    fn = tc.function
                    name = fn.name
                    args_raw = fn.arguments
                    tid = getattr(tc, "id", None)
                try:
                    argd = (
                        json.loads(args_raw)
                        if isinstance(args_raw, str)
                        else (args_raw or {})
                    )
                except json.JSONDecodeError:
                    argd = {}
                if not isinstance(argd, dict):
                    argd = {"value": argd}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": _tool_call_id(tid),
                        "name": name,
                        "input": argd,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            anth_msgs.append({"role": "assistant", "content": blocks})
            i += 1
            continue
        if role == "tool":
            result_blocks: List[Dict[str, Any]] = []
            while i < n and messages[i].get("role") == "tool":
                tm = messages[i]
                raw = tm.get("content", "")
                if isinstance(raw, (dict, list)):
                    content_str = json.dumps(raw)
                else:
                    content_str = str(raw)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(tm.get("tool_call_id") or ""),
                        "content": content_str,
                    }
                )
                i += 1
            anth_msgs.append({"role": "user", "content": result_blocks})
            continue
        i += 1

    system_text = "\n\n".join(system_chunks) if system_chunks else None
    return system_text, _merge_consecutive_user_messages(anth_msgs)


def _merge_consecutive_user_messages(
    msgs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Anthropic expects alternating user/assistant turns; merge adjacent user messages."""
    if not msgs:
        return msgs
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if m.get("role") != "user":
            out.append(m)
            continue
        c = m.get("content")
        if out and out[-1].get("role") == "user":
            prev = out[-1].get("content")
            if isinstance(prev, str) and isinstance(c, str):
                out[-1] = {"role": "user", "content": f"{prev}\n\n{c}"}
            elif isinstance(prev, list) and isinstance(c, list):
                out[-1] = {"role": "user", "content": list(prev) + list(c)}
            elif isinstance(prev, str) and isinstance(c, list):
                out[-1] = {
                    "role": "user",
                    "content": [{"type": "text", "text": prev}, *c],
                }
            elif isinstance(prev, list) and isinstance(c, str):
                out[-1] = {
                    "role": "user",
                    "content": [*prev, {"type": "text", "text": c}],
                }
            else:
                out.append(m)
        else:
            out.append(m)
    return out


def _tighten_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort root object strictness for Anthropic structured outputs."""
    out = dict(schema)
    if out.get("type") == "object" and "additionalProperties" not in out:
        out = {**out, "additionalProperties": False}
    return out


def _map_anthropic_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, RateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        if code == 429:
            return ProviderRateLimitError(str(exc))
        body = str(exc).lower()
        if code is not None and 400 <= code < 500:
            if any(
                x in body
                for x in (
                    "content_policy",
                    "safety",
                    "moderation",
                    "blocked",
                    "harm",
                )
            ):
                return ProviderContentPolicyError(str(exc))
        return ProviderAPIError(str(exc), status_code=code)
    if isinstance(exc, AnthropicAPIError):
        code = getattr(exc, "status_code", None)
        return ProviderAPIError(str(exc), status_code=code)
    return exc


def anthropic_message_to_chat_result(
    message: Any,
    *,
    response_model_cls: Optional[Type[BaseModel]] = None,
) -> ChatCompletionResult:
    """Build ``ChatCompletionResult`` from an Anthropic ``Message`` (sync parse)."""
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    text_parts: List[str] = []
    tool_calls: List[Any] = []
    for block in message.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t)
        elif btype == "tool_use":
            bid = getattr(block, "id", None) or _tool_call_id(None)
            name = getattr(block, "name", "") or ""
            inp = getattr(block, "input", None)
            if not isinstance(inp, dict):
                inp = {}
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=str(bid),
                    type="function",
                    function=Function(
                        name=name,
                        arguments=json.dumps(inp),
                    ),
                )
            )

    text = "".join(text_parts).strip() or ""
    parsed_out: Any = None
    if response_model_cls is not None and text:
        try:
            parsed_out = response_model_cls.model_validate_json(text)
        except Exception as e:
            logger.warning("Anthropic structured output parse failed: %s", e)

    usage_obj = getattr(message, "usage", None)
    prompt_t = int(getattr(usage_obj, "input_tokens", None) or 0) if usage_obj else 0
    completion_t = int(getattr(usage_obj, "output_tokens", None) or 0) if usage_obj else 0
    usage = _UsageStats(
        prompt_tokens=prompt_t,
        completion_tokens=completion_t,
        total_tokens=prompt_t + completion_t,
    )

    finish = getattr(message, "stop_reason", None)
    msg = _ChatMessage(
        content=text if text else None,
        tool_calls=tool_calls if tool_calls else None,
        parsed=parsed_out,
    )
    return ChatCompletionResult(
        choices=[_ChatChoice(message=msg, finish_reason=str(finish) if finish else None)],
        usage=usage,
    )


def _prepare_extra_kwargs(api_params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in api_params.items():
        if k in _SKIP_KEYS or v is None:
            continue
        if k in _ALLOWED_MESSAGE_KWARGS:
            out[k] = v
    return out


async def anthropic_chat_completion(
    *,
    profile: ModelProfile,
    messages: List[Dict[str, Any]],
    api_params: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    timeout: float,
) -> ChatCompletionResult:
    model_id = _strip_model_id(profile.model)
    params = {k: v for k, v in api_params.items() if k not in _SKIP_KEYS and v is not None}
    response_model_cls: Optional[Type[BaseModel]] = None
    rf = params.pop("response_format", None)
    if isinstance(rf, type) and issubclass(rf, BaseModel):
        response_model_cls = cast(Type[BaseModel], rf)

    max_tokens = params.pop("max_tokens", None)
    if max_tokens is None:
        raise ProviderAPIError(
            "Anthropic messages API requires max_tokens",
            status_code=None,
        )
    max_tokens = int(max_tokens)

    output_config: Optional[Dict[str, Any]] = None
    if response_model_cls is not None:
        schema = _tighten_json_schema(response_model_cls.model_json_schema())
        output_config = {
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        }

    anthropic_tools = openai_tools_to_anthropic(tools)
    system_text, anthropic_messages = openai_messages_to_anthropic(messages)
    if not anthropic_messages:
        raise ProviderAPIError(
            "Anthropic requires at least one user/assistant/tool message in the conversation.",
            status_code=None,
        )

    extra = _prepare_extra_kwargs(params)

    client = AsyncAnthropic(
        api_key=profile.api_key or "",
        base_url=profile.api_base,
        timeout=timeout,
        max_retries=0,
    )

    create_kwargs: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        **extra,
    }
    if system_text:
        create_kwargs["system"] = system_text
    if anthropic_tools:
        create_kwargs["tools"] = anthropic_tools
    if output_config is not None:
        create_kwargs["output_config"] = output_config

    try:
        message = await client.messages.create(**create_kwargs)
    except (APIStatusError, RateLimitError, AnthropicAPIError) as e:
        raise _map_anthropic_exception(e) from e

    return anthropic_message_to_chat_result(
        message, response_model_cls=response_model_cls
    )
