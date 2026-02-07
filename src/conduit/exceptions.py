from __future__ import annotations

from typing import Any


class ConduitError(Exception):
    """Base exception for Conduit."""


class ConfigValidationError(ConduitError):
    """Raised when configuration or overrides fail validation."""


class ProviderError(ConduitError):
    """Raised for provider HTTP and transport failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.details = details


class AuthenticationError(ProviderError):
    """Raised when credentials are missing or invalid."""


class RateLimitError(ProviderError):
    """Raised when provider rate limits a request."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        status_code: int | None = None,
        provider: str | None = None,
        details: Any | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            provider=provider,
            details=details,
        )
        self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
    """Raised when a provider cannot find the requested model."""


class ContextLengthError(ProviderError):
    """Raised when prompt or completion exceeds context limits."""


class ProviderUnavailableError(ProviderError):
    """Raised when provider service is unavailable."""


class ResponseParseError(ProviderError):
    """Raised when a provider response cannot be parsed."""


class ToolError(ConduitError):
    """Base exception for tool schema and parsing failures."""


class ToolSchemaError(ToolError):
    """Raised when a tool definition is invalid."""


class ToolCallParseError(ToolError):
    """Raised when parsing tool calls fails."""


class StreamError(ConduitError):
    """Raised when a stream fails or returns malformed data."""
