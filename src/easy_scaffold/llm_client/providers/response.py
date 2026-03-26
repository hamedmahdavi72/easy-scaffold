"""Unified chat completion shape for Gemini-built responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class _CompletionTokensDetails:
    reasoning_tokens: int = 0
    text_tokens: int = 0


@dataclass
class _UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: Optional[_CompletionTokensDetails] = None


@dataclass
class _ChatMessage:
    content: Optional[str]
    tool_calls: Optional[List[Any]] = None
    parsed: Optional[Any] = None

    def model_dump(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls is not None:
            tcs = self.tool_calls
            if tcs and hasattr(tcs[0], "model_dump"):
                d["tool_calls"] = [tc.model_dump() for tc in tcs]
            elif tcs and hasattr(tcs[0], "dict"):
                d["tool_calls"] = [tc.dict() for tc in tcs]
            else:
                d["tool_calls"] = tcs
        return d


@dataclass
class _ChatChoice:
    message: _ChatMessage
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletionResult:
    choices: List[_ChatChoice] = field(default_factory=list)
    usage: _UsageStats = field(default_factory=_UsageStats)
