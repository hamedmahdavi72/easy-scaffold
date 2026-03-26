"""OpenAI API and OpenAI-compatible endpoints (DeepSeek)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, cast

from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI, OpenAIError, RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel

from ...configs.pydantic_models import ModelProfile
from .errors import ProviderAPIError, ProviderContentPolicyError, ProviderRateLimitError

logger = logging.getLogger(__name__)

_SKIP_KEYS = frozenset(
    {
        "allowed_openai_params",
        "max_output_tokens",
        "thinking",
        "response_mime_type",
    }
)


def _strip_litellm_model_id(model: str) -> str:
    for prefix in ("openai/", "deepseek/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _prepare_params(api_params: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in api_params.items() if k not in _SKIP_KEYS and v is not None}
    return out





def _openai_messages_include_image_parts(
    messages: List[Dict[str, Any]],
) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


def _map_openai_exception(exc: Exception) -> Exception:
    if isinstance(exc, OpenAIRateLimitError):
        return ProviderRateLimitError(str(exc))
    if isinstance(exc, OpenAIAPIError):
        code = getattr(exc, "status_code", None)
        body = str(exc).lower()
        if code == 429:
            return ProviderRateLimitError(str(exc))
        if code is not None and 400 <= code < 500:
            if any(
                x in body
                for x in ("content_policy", "safety", "moderation", "blocked")
            ):
                return ProviderContentPolicyError(str(exc))
        return ProviderAPIError(str(exc), status_code=code)
    if isinstance(exc, OpenAIError):
        code = getattr(exc, "status_code", None)
        return ProviderAPIError(str(exc), status_code=code)
    return exc


async def openai_chat_completion(
    *,
    profile: ModelProfile,
    messages: List[Dict[str, Any]],
    api_params: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    timeout: float,
) -> Any:
    model_id = _strip_litellm_model_id(profile.model)
    base_url = profile.api_base
    client = AsyncOpenAI(
        api_key=profile.api_key or "",
        base_url=base_url,
        timeout=timeout,
    )
    params = _prepare_params(dict(api_params))
    rf = params.pop("response_format", None)

    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        **params,
    }
    if tools:
        kwargs["tools"] = tools

    try:
        use_parse = (
            isinstance(rf, type)
            and issubclass(rf, BaseModel)
            and not _openai_messages_include_image_parts(messages)
        )
        if use_parse:
            resp = await client.chat.completions.parse(
                response_format=cast(Type[BaseModel], rf),
                **kwargs,
            )
        else:
            resp = await client.chat.completions.create(**kwargs)
        return resp
    except (OpenAIAPIError, OpenAIRateLimitError, OpenAIError) as e:
        raise _map_openai_exception(e) from e
    except Exception:
        raise



@dataclass
class ImageGenerationResult:
    image_bytes: bytes
    mime_type: str
    revised_prompt: str | None = None


async def openai_generate_image(
    *,
    profile: ModelProfile,
    prompt: str,
    api_params: Dict[str, Any],
    timeout: float,
) -> ImageGenerationResult:
    import base64

    model_id = _strip_litellm_model_id(profile.model)
    client = AsyncOpenAI(
        api_key=profile.api_key or "",
        base_url=profile.api_base,
        timeout=timeout,
    )
    params = dict(api_params)
    params.pop("response_format", None)
    params.pop("max_tokens", None)
    size = str(params.pop("image_size", "1024x1024"))
    quality = str(params.pop("image_quality", "standard"))
    n = int(params.pop("n", 1))
    try:
        resp = await client.images.generate(
            model=model_id,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            response_format="b64_json",
        )
    except (OpenAIAPIError, OpenAIRateLimitError, OpenAIError) as e:
        raise _map_openai_exception(e) from e
    item = resp.data[0]
    b64 = getattr(item, "b64_json", None)
    if not b64:
        raise ProviderAPIError("OpenAI image response missing b64_json", status_code=None)
    raw = base64.b64decode(b64)
    rp = getattr(item, "revised_prompt", None)
    return ImageGenerationResult(image_bytes=raw, mime_type="image/png", revised_prompt=rp)
