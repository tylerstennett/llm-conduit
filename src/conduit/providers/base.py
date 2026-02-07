from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

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
from conduit.models.messages import ChatRequest, ChatResponse, ChatResponseChunk, Message, Role, UsageStats
from conduit.tools.schema import ToolCall, ToolDefinition, parse_tool_arguments


class BaseProvider(ABC):
    """Base provider abstraction."""

    provider_name: str

    def __init__(
        self,
        config: BaseLLMConfig,
        *,
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.timeout = timeout
        self._client_owned = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if self._client_owned:
            await self.http_client.aclose()

    @abstractmethod
    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
        stream: bool,
    ) -> dict[str, Any]:
        """Build provider-native request payload."""

    @abstractmethod
    async def chat(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
    ) -> ChatResponse:
        """Execute non-streaming chat."""

    @abstractmethod
    async def chat_stream(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
    ) -> AsyncIterator[ChatResponseChunk]:
        """Execute streaming chat."""

    def default_headers(
        self,
        *,
        effective_config: BaseLLMConfig | None = None,
    ) -> dict[str, str]:
        config = effective_config if effective_config is not None else self.config
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    def make_url(
        self,
        path: str,
        *,
        effective_config: BaseLLMConfig | None = None,
    ) -> str:
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
        effective_config: BaseLLMConfig | None = None,
    ) -> dict[str, Any]:
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
        if response.status_code < 400:
            return
        await response.aread()
        raise self.map_http_error(response)

    def map_http_error(self, response: httpx.Response) -> ProviderError:
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


def normalize_stop(stop: list[str] | str | None) -> list[str] | str | None:
    if stop is None:
        return None
    if isinstance(stop, list):
        return stop if stop else None
    return stop


def tool_choice_to_openai_payload(
    tool_choice: str | dict[str, Any] | None,
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise ConfigValidationError(
                "tool_choice string must be one of: auto, none, required"
            )
        return tool_choice
    if isinstance(tool_choice, dict):
        return tool_choice
    raise ConfigValidationError("tool_choice must be a string, dict, or None")


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        payload: dict[str, Any] = {"role": message.role.value}
        if message.name:
            payload["name"] = message.name

        if message.content is not None:
            payload["content"] = message.content

        if message.images:
            if message.content is None:
                payload["content"] = ""
            if isinstance(payload["content"], str):
                content_items: list[dict[str, Any]] = []
                if payload["content"]:
                    content_items.append({"type": "text", "text": payload["content"]})
                for image in message.images:
                    content_items.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image},
                        }
                    )
                payload["content"] = content_items

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]

        if message.role is Role.TOOL and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id

        output.append(payload)
    return output


def tool_definitions_to_openai(
    tools: list[ToolDefinition] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [tool.to_openai_tool() for tool in tools]


def parse_openai_tool_calls(
    raw_calls: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    if not raw_calls:
        return None

    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        function = raw_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        arguments = parse_tool_arguments(function.get("arguments"))
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))

    return parsed or None


def parse_usage(usage: dict[str, Any] | None) -> UsageStats | None:
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return UsageStats(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=(
            completion_tokens if isinstance(completion_tokens, int) else None
        ),
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )
