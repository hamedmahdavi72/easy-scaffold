"""Dispatch chat completions to provider implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...configs.pydantic_models import ModelProfile
from .errors import ProviderAPIError
from .anthropic import anthropic_chat_completion
from .gemini import gemini_chat_completion
from .openai_compat import (
    ImageGenerationResult,
    openai_chat_completion,
    openai_generate_image,
)


async def route_chat_completion(
    *,
    profile: ModelProfile,
    messages: List[Dict[str, Any]],
    api_params: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    timeout: float,
) -> Any:
    if profile.provider == "gemini":
        return await gemini_chat_completion(
            profile=profile,
            messages=messages,
            api_params=api_params,
            tools=tools,
            timeout=timeout,
        )
    if profile.provider in ("openai", "deepseek"):
        return await openai_chat_completion(
            profile=profile,
            messages=messages,
            api_params=api_params,
            tools=tools,
            timeout=timeout,
        )
    if profile.provider == "anthropic":
        return await anthropic_chat_completion(
            profile=profile,
            messages=messages,
            api_params=api_params,
            tools=tools,
            timeout=timeout,
        )
    raise ProviderAPIError(
        f"Unsupported provider for router: {profile.provider!r}",
        status_code=None,
    )



async def route_image_generation(
    *,
    profile: ModelProfile,
    prompt: str,
    api_params: Dict[str, Any],
    timeout: float,
) -> ImageGenerationResult:
    if profile.provider == "openai":
        return await openai_generate_image(
            profile=profile,
            prompt=prompt,
            api_params=api_params,
            timeout=timeout,
        )
    raise ProviderAPIError(
        f"Unsupported provider for image generation: {profile.provider!r}",
        status_code=None,
    )
