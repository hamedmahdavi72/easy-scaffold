"""Gemini chat completions via google-genai (async)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Type, cast

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel

from ...configs.pydantic_models import ModelProfile
from .errors import ProviderAPIError, ProviderContentPolicyError, ProviderRateLimitError
from .vision_utils import parse_data_url_image
from .response import (
    ChatCompletionResult,
    _ChatChoice,
    _ChatMessage,
    _CompletionTokensDetails,
    _UsageStats,
)

logger = logging.getLogger(__name__)

def _safe_http_code(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None



def _strip_model_id(model: str) -> str:
    if model.startswith("gemini/"):
        return model[7:]
    return model


def _tool_call_id(raw: Any) -> str:
    if raw:
        return str(raw)
    return f"call_{uuid.uuid4().hex[:24]}"


def _openai_tools_to_genai(tools: List[Dict[str, Any]]) -> Optional[List[types.Tool]]:
    decls: List[types.FunctionDeclaration] = []
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
        decls.append(
            types.FunctionDeclaration(
                name=name,
                description=fn.get("description") or "",
                parameters_json_schema=params,
            )
        )
    if not decls:
        return None
    return [types.Tool(function_declarations=decls)]


def _openai_messages_to_contents(
    messages: List[Dict[str, Any]],
) -> tuple[Optional[str], List[types.Content]]:
    system_chunks: List[str] = []
    out: List[types.Content] = []
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
            c = m.get("content")
            parts_u: List[types.Part] = []
            if isinstance(c, str):
                parts_u.append(types.Part(text=c))
            elif isinstance(c, list):
                for block in c:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")
                    if bt == "text":
                        parts_u.append(
                            types.Part(text=str(block.get("text") or ""))
                        )
                    elif bt == "image_url":
                        iu = block.get("image_url") or {}
                        url = iu.get("url") if isinstance(iu, dict) else None
                        if not url:
                            continue
                        parsed = parse_data_url_image(str(url))
                        if parsed:
                            raw_b, mime = parsed
                            parts_u.append(
                                types.Part.from_bytes(data=raw_b, mime_type=mime)
                            )
            else:
                parts_u.append(
                    types.Part(text=str(c) if c is not None else "")
                )
            if not parts_u:
                parts_u.append(types.Part(text=""))
            out.append(types.Content(role="user", parts=parts_u))
            i += 1
            continue
        if role in ("assistant", "model"):
            parts: List[types.Part] = []
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(types.Part(text=c))
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
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=name,
                            args=argd,
                            id=_tool_call_id(tid),
                        )
                    )
                )
            if not parts:
                parts.append(types.Part(text=""))
            out.append(types.Content(role="model", parts=parts))
            i += 1
            continue
        if role == "tool":
            fr_parts: List[types.Part] = []
            while i < n and messages[i].get("role") == "tool":
                tm = messages[i]
                name = tm.get("name") or ""
                raw = tm.get("content", "")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    parsed = {"result": raw}
                if not isinstance(parsed, dict):
                    parsed = {"result": parsed}
                fr_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response=parsed,
                            id=tm.get("tool_call_id"),
                        )
                    )
                )
                i += 1
            out.append(types.Content(role="user", parts=fr_parts))
            continue
        i += 1

    system_text = "\n\n".join(system_chunks) if system_chunks else None
    return system_text, out


def _map_genai_error(exc: BaseException) -> BaseException:
    if isinstance(exc, genai_errors.ClientError):
        code = _safe_http_code(getattr(exc, "code", None))
        if code == 429:
            return ProviderRateLimitError(str(exc))
        msg = str(exc).lower()
        if code is not None and 400 <= code < 500:
            if any(
                k in msg
                for k in (
                    "safety",
                    "blocked",
                    "policy",
                    "content",
                    "harm",
                )
            ):
                return ProviderContentPolicyError(str(exc))
        return ProviderAPIError(str(exc), status_code=code)
    if isinstance(exc, genai_errors.ServerError):
        code = _safe_http_code(getattr(exc, "code", None))
        return ProviderAPIError(str(exc), status_code=code if code is not None else 500)
    if isinstance(exc, genai_errors.APIError):
        code = _safe_http_code(getattr(exc, "code", None))
        return ProviderAPIError(str(exc), status_code=code)
    return exc


def _usage_from_metadata(
    meta: Optional[types.GenerateContentResponseUsageMetadata],
    text_out_tokens: int,
    reasoning_tokens: int,
) -> _UsageStats:
    if meta is None:
        return _UsageStats(
            prompt_tokens=0,
            completion_tokens=text_out_tokens + reasoning_tokens,
            total_tokens=text_out_tokens + reasoning_tokens,
            completion_tokens_details=_CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                text_tokens=text_out_tokens,
            ),
        )
    prompt = int(meta.prompt_token_count or 0)
    candidates = int(meta.candidates_token_count or 0)
    thoughts = int(meta.thoughts_token_count or 0)
    total = int(meta.total_token_count or (prompt + candidates))
    r = thoughts
    t = max(0, candidates - thoughts) if candidates else text_out_tokens
    if text_out_tokens and t == 0 and not reasoning_tokens:
        t = text_out_tokens
    return _UsageStats(
        prompt_tokens=prompt,
        completion_tokens=candidates,
        total_tokens=total,
        completion_tokens_details=_CompletionTokensDetails(
            reasoning_tokens=r,
            text_tokens=t,
        ),
    )


def _candidate_to_message(
    cand: Optional[types.Candidate],
) -> tuple[str, List[Any], Optional[str]]:
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    text_parts: List[str] = []
    tool_calls: List[Any] = []
    if cand is None or not cand.content or not cand.content.parts:
        return "", [], None
    for part in cand.content.parts:
        if part.thought:
            continue
        if part.text:
            text_parts.append(part.text)
        if part.function_call:
            fc = part.function_call
            tid = _tool_call_id(getattr(fc, "id", None))
            args = fc.args if isinstance(fc.args, dict) else {}
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    id=tid,
                    type="function",
                    function=Function(
                        name=fc.name or "",
                        arguments=json.dumps(args),
                    ),
                )
            )
    finish = None
    if cand.finish_reason:
        finish = str(cand.finish_reason)
    return ("".join(text_parts).strip() or ""), tool_calls, finish


async def gemini_chat_completion(
    *,
    profile: ModelProfile,
    messages: List[Dict[str, Any]],
    api_params: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    timeout: float,
) -> ChatCompletionResult:
    model_id = _strip_model_id(profile.model)
    api_key = profile.api_key or ""
    timeout_ms = max(1, int(timeout * 1000))
    http_options = types.HttpOptions(timeout=timeout_ms)

    client = genai.Client(api_key=api_key, http_options=http_options)
    aio = client.aio

    system_text, contents = _openai_messages_to_contents(messages)

    params = dict(api_params)
    response_model_cls: Optional[Type[BaseModel]] = None
    rf = params.get("response_format")
    if isinstance(rf, type) and issubclass(rf, BaseModel):
        response_model_cls = cast(Type[BaseModel], rf)

    temperature = params.get("temperature")
    max_out = params.get("max_output_tokens")
    thinking_raw = params.get("thinking")
    mime = params.get("response_mime_type")
    top_p = params.get("top_p")
    top_k = params.get("top_k")

    thinking_cfg: Optional[types.ThinkingConfig] = None
    if isinstance(thinking_raw, dict):
        enabled = thinking_raw.get("type") == "enabled"
        budget = thinking_raw.get("budget_tokens")
        thinking_cfg = types.ThinkingConfig(
            include_thoughts=enabled,
            thinking_budget=int(budget) if budget is not None else None,
        )

    gen_tools = _openai_tools_to_genai(tools or [])
    afc = None
    if gen_tools:
        afc = types.AutomaticFunctionCallingConfig(disable=True)

    json_schema: Optional[Any] = None
    if response_model_cls is not None:
        json_schema = response_model_cls.model_json_schema()
        if not mime:
            mime = "application/json"

    config_kwargs: Dict[str, Any] = {}
    if system_text:
        config_kwargs["system_instruction"] = system_text
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if max_out is not None:
        config_kwargs["max_output_tokens"] = int(max_out)
    if top_p is not None:
        config_kwargs["top_p"] = float(top_p)
    if top_k is not None:
        config_kwargs["top_k"] = float(top_k)
    if thinking_cfg is not None:
        config_kwargs["thinking_config"] = thinking_cfg
    if mime:
        config_kwargs["response_mime_type"] = mime
    if json_schema is not None:
        config_kwargs["response_json_schema"] = json_schema
    if gen_tools:
        config_kwargs["tools"] = gen_tools
    if afc is not None:
        config_kwargs["automatic_function_calling"] = afc

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    try:
        resp = await aio.models.generate_content(
            model=model_id,
            contents=contents,
            config=config,
        )
    except (genai_errors.ClientError, genai_errors.ServerError, genai_errors.APIError) as e:
        raise _map_genai_error(e) from e

    if resp.prompt_feedback and resp.prompt_feedback.block_reason:
        raise ProviderContentPolicyError(
            f"Prompt blocked: {resp.prompt_feedback.block_reason}"
        )

    cand = resp.candidates[0] if resp.candidates else None
    text, tool_calls, finish = _candidate_to_message(cand)

    parsed_out: Any = None
    if response_model_cls is not None and text:
        try:
            parsed_out = response_model_cls.model_validate_json(text)
        except Exception as e:
            logger.warning("Gemini structured output parse failed: %s", e)

    meta = resp.usage_metadata
    thoughts = int(meta.thoughts_token_count or 0) if meta else 0
    cand_tokens = int(meta.candidates_token_count or 0) if meta else 0
    text_tokens = max(0, cand_tokens - thoughts)
    usage = _usage_from_metadata(meta, text_tokens, thoughts)

    msg = _ChatMessage(
        content=text if text else None,
        tool_calls=tool_calls if tool_calls else None,
        parsed=parsed_out,
    )
    result = ChatCompletionResult(
        choices=[_ChatChoice(message=msg, finish_reason=finish)],
        usage=usage,
    )
    return result
