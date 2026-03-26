"""Exceptions raised by provider adapters; mapped in LLM client.create()."""


class ProviderRateLimitError(Exception):
    """HTTP 429 or provider-specific quota exhaustion."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message)
        self.status_code = 429


class ProviderContentPolicyError(Exception):
    """Content blocked by provider safety or policy."""


class ProviderAPIError(Exception):
    """Non-rate-limit API failure (4xx/5xx or unknown)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
