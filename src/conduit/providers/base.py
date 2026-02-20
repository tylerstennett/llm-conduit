from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, TypeVar

import httpx

from conduit.config.base import BaseLLMConfig
from conduit.exceptions import (
    AuthenticationError,
    ConfigValidationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from conduit.models.messages import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    Message,
)


ConfigT = TypeVar("ConfigT", bound=BaseLLMConfig)


class BaseProvider(ABC, Generic[ConfigT]):
    """Base provider abstraction.

    Concrete providers implement :meth:`build_request_body`, :meth:`chat`,
    and :meth:`chat_stream`.  The base class supplies shared HTTP helpers
    and error mapping.

    Args:
        config: Provider-specific configuration.
        timeout: HTTP request timeout in seconds.
        http_client: Optional pre-configured ``httpx.AsyncClient``.  When
            ``None`` (the default) a new client is created and owned by this
            provider instance.
    """

    provider_name: str
    supported_runtime_override_keys: frozenset[str] = frozenset()

    def __init__(
        self,
        config: ConfigT,
        *,
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self._client_owned = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the HTTP client if it is owned by this provider."""
        if self._client_owned:
            await self.http_client.aclose()

    @abstractmethod
    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: ConfigT,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the provider-native HTTP request payload.

        Args:
            request: Canonical chat request.
            effective_config: Resolved provider config (with overrides applied).
            stream: Whether to request streaming output.

        Returns:
            JSON-serialisable request body dict.
        """

    @abstractmethod
    async def chat(
        self,
        request: ChatRequest,
        *,
        effective_config: ConfigT,
    ) -> ChatResponse:
        """Execute a non-streaming chat completion.

        Args:
            request: Canonical chat request.
            effective_config: Resolved provider config.

        Returns:
            The complete chat response.
        """

    @abstractmethod
    async def chat_stream(
        self,
        request: ChatRequest,
        *,
        effective_config: ConfigT,
    ) -> AsyncIterator[ChatResponseChunk]:
        """Execute a streaming chat completion.

        Args:
            request: Canonical chat request.
            effective_config: Resolved provider config.

        Yields:
            ChatResponseChunk: Individual stream chunks.
        """

    def default_headers(
        self,
        *,
        effective_config: ConfigT | None = None,
    ) -> dict[str, str]:
        """Build default HTTP headers including authorization.

        Args:
            effective_config: Config to read ``api_key`` from.  Falls back
                to ``self.config`` when ``None``.

        Returns:
            Mutable header dict.
        """
        config = effective_config if effective_config is not None else self.config
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    def make_url(
        self,
        path: str,
        *,
        effective_config: ConfigT | None = None,
    ) -> str:
        """Resolve a relative API path to a full URL.

        Absolute URLs (starting with ``http://`` or ``https://``) are
        returned unchanged.

        Args:
            path: Relative or absolute endpoint path.
            effective_config: Config to read ``base_url`` from.  Falls
                back to ``self.config`` when ``None``.

        Returns:
            Fully-qualified URL string.
        """
        config = effective_config if effective_config is not None else self.config
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{config.base_url.rstrip('/')}/{path.lstrip('/')}"

    async def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        effective_config: ConfigT | None = None,
    ) -> dict[str, Any]:
        """POST *payload* as JSON and return the parsed response dict.

        Raises a ``ProviderError`` subtype on HTTP errors or non-JSON
        responses.

        Args:
            path: Relative or absolute endpoint path.
            payload: JSON-serialisable request body.
            headers: Extra headers merged on top of :meth:`default_headers`.
            effective_config: Config for URL/header resolution.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            ProviderUnavailableError: On transport-level failures.
            ProviderError: On HTTP 4xx/5xx responses.
        """
        merged_headers = self.default_headers(effective_config=effective_config)
        if headers:
            merged_headers.update(headers)
        url = self.make_url(path, effective_config=effective_config)

        try:
            response = await self.http_client.post(url, json=payload, headers=merged_headers)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self.provider_name} request failed: {exc}",
                provider=self.provider_name,
            ) from exc

        if response.status_code >= 400:
            raise self.map_http_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.provider_name} returned non-JSON response",
                status_code=response.status_code,
                provider=self.provider_name,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                f"{self.provider_name} returned unexpected JSON payload",
                status_code=response.status_code,
                provider=self.provider_name,
                details=data,
            )
        return data

    async def raise_for_stream_status(self, response: httpx.Response) -> None:
        """Raise a mapped ``ProviderError`` if the stream response is an error.

        Args:
            response: The ``httpx.Response`` from a streaming request.

        Raises:
            ProviderError: On HTTP 4xx/5xx status codes.
        """
        if response.status_code < 400:
            return
        await response.aread()
        raise self.map_http_error(response)

    def map_http_error(self, response: httpx.Response) -> ProviderError:
        """Map an HTTP error response to a ``ProviderError`` subtype.

        Status codes are mapped as follows: 401/403 to
        ``AuthenticationError``, 429 to ``RateLimitError``, 404 (with
        "model" in the message) to ``ModelNotFoundError``, 400 (context
        length) to ``ContextLengthError``, 5xx to
        ``ProviderUnavailableError``, and everything else to the base
        ``ProviderError``.

        Args:
            response: The failed ``httpx.Response``.

        Returns:
            An appropriate ``ProviderError`` subclass instance.
        """
        message = self.extract_error_message(response)
        status_code = response.status_code
        details: Any | None = None
        try:
            details = response.json()
        except ValueError:
            details = response.text

        lower_message = message.lower()
        if status_code in {401, 403}:
            return AuthenticationError(
                message,
                status_code=status_code,
                provider=self.provider_name,
                details=details,
            )
        if status_code == 429:
            retry_after_header = response.headers.get("Retry-After")
            retry_after = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
            return RateLimitError(
                message,
                retry_after=retry_after,
                status_code=status_code,
                provider=self.provider_name,
                details=details,
            )
        if status_code == 404 and "model" in lower_message:
            return ModelNotFoundError(
                message,
                status_code=status_code,
                provider=self.provider_name,
                details=details,
            )
        if status_code == 400 and "context" in lower_message and "length" in lower_message:
            return ContextLengthError(
                message,
                status_code=status_code,
                provider=self.provider_name,
                details=details,
            )
        if status_code >= 500:
            return ProviderUnavailableError(
                message,
                status_code=status_code,
                provider=self.provider_name,
                details=details,
            )
        return ProviderError(
            message,
            status_code=status_code,
            provider=self.provider_name,
            details=details,
        )

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
        """Extract a human-readable error message from a failed response.

        Attempts to read ``error.message``, ``error`` (string), or
        ``message`` from the JSON body, falling back to the raw response
        text.

        Args:
            response: The failed ``httpx.Response``.

        Returns:
            Error message string.
        """
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = error_obj.get("message")
                if isinstance(message, str) and message.strip():
                    return message
            if isinstance(error_obj, str) and error_obj.strip():
                return error_obj
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message

        text = response.text.strip()
        if text:
            return text
        return f"provider request failed with status {response.status_code}"
