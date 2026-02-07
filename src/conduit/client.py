from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, TypeVar

from pydantic import ValidationError

from conduit.config import BaseLLMConfig, OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.models.messages import ChatRequest, ChatResponse, ChatResponseChunk, Message
from conduit.providers import BaseProvider, OllamaProvider, OpenRouterProvider, VLLMProvider
from conduit.retry import RetryPolicy
from conduit.tools.schema import ToolDefinition

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    config_type: type[BaseLLMConfig]
    provider_type: type[BaseProvider]
    config_from_env: Callable[[], BaseLLMConfig]


def _build_vllm_config_from_env() -> BaseLLMConfig:
    model = os.getenv("CONDUIT_VLLM_MODEL") or os.getenv("VLLM_MODEL")
    if not model:
        raise ConfigValidationError(
            "missing CONDUIT_VLLM_MODEL or VLLM_MODEL environment variable"
        )
    return VLLMConfig(
        model=model,
        base_url=os.getenv("CONDUIT_VLLM_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("CONDUIT_VLLM_API_KEY"),
    )


def _build_ollama_config_from_env() -> BaseLLMConfig:
    model = os.getenv("CONDUIT_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")
    if not model:
        raise ConfigValidationError(
            "missing CONDUIT_OLLAMA_MODEL or OLLAMA_MODEL environment variable"
        )
    return OllamaConfig(
        model=model,
        base_url=os.getenv("CONDUIT_OLLAMA_URL", "http://localhost:11434"),
        api_key=os.getenv("CONDUIT_OLLAMA_API_KEY"),
    )


def _build_openrouter_config_from_env() -> BaseLLMConfig:
    model = os.getenv("CONDUIT_OPENROUTER_MODEL")
    api_key = os.getenv("CONDUIT_OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not model:
        raise ConfigValidationError("missing CONDUIT_OPENROUTER_MODEL")
    if not api_key:
        raise ConfigValidationError(
            "missing CONDUIT_OPENROUTER_KEY or OPENROUTER_API_KEY"
        )
    return OpenRouterConfig(
        model=model,
        api_key=api_key,
        base_url=os.getenv(
            "CONDUIT_OPENROUTER_URL",
            "https://openrouter.ai/api/v1",
        ),
        app_name=os.getenv("CONDUIT_OPENROUTER_APP_NAME"),
        app_url=os.getenv("CONDUIT_OPENROUTER_APP_URL"),
    )


_PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="vllm",
        config_type=VLLMConfig,
        provider_type=VLLMProvider,
        config_from_env=_build_vllm_config_from_env,
    ),
    ProviderSpec(
        name="ollama",
        config_type=OllamaConfig,
        provider_type=OllamaProvider,
        config_from_env=_build_ollama_config_from_env,
    ),
    ProviderSpec(
        name="openrouter",
        config_type=OpenRouterConfig,
        provider_type=OpenRouterProvider,
        config_from_env=_build_openrouter_config_from_env,
    ),
)
_PROVIDER_SPECS_BY_NAME = {spec.name: spec for spec in _PROVIDER_SPECS}


def _supported_provider_names() -> str:
    return ", ".join(spec.name for spec in _PROVIDER_SPECS)


def _provider_spec_for_config(config: BaseLLMConfig) -> ProviderSpec:
    for spec in _PROVIDER_SPECS:
        if isinstance(config, spec.config_type):
            return spec
    raise ConfigValidationError(f"unsupported config type: {type(config)!r}")


class Conduit:
    """Provider-aware asynchronous chat client."""

    def __init__(
        self,
        config: BaseLLMConfig,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.timeout = timeout
        self._provider = self._create_provider(config)

    async def __aenter__(self) -> "Conduit":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        request = ChatRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            config_overrides=config_overrides,
        )
        effective_config = self._apply_overrides(config_overrides)

        if stream:
            return await self._aggregate_stream_response(request, effective_config)

        return await self._chat_with_retry(request, effective_config)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatResponseChunk]:
        request = ChatRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            config_overrides=config_overrides,
        )
        effective_config = self._apply_overrides(config_overrides)

        if self.retry_policy is None:
            async for chunk in self._provider.chat_stream(
                request,
                effective_config=effective_config,
            ):
                yield chunk
            return

        attempt = 0
        while True:
            produced_output = False
            try:
                async for chunk in self._provider.chat_stream(
                    request,
                    effective_config=effective_config,
                ):
                    produced_output = True
                    yield chunk
                return
            except BaseException as exc:
                if produced_output:
                    raise
                attempt += 1
                if not self.retry_policy.should_retry(exc, attempt):
                    raise
                await asyncio.sleep(self.retry_policy.backoff_for_attempt(attempt, exc))

    @staticmethod
    def from_env(provider: str = "openrouter") -> "Conduit":
        normalized = provider.strip().lower()
        spec = _PROVIDER_SPECS_BY_NAME.get(normalized)
        if spec is None:
            raise ConfigValidationError(
                f"provider must be one of: {_supported_provider_names()}"
            )
        return Conduit(spec.config_from_env())

    async def _chat_with_retry(
        self,
        request: ChatRequest,
        effective_config: BaseLLMConfig,
    ) -> ChatResponse:
        if self.retry_policy is None:
            return await self._provider.chat(request, effective_config=effective_config)

        attempt = 0
        while True:
            try:
                return await self._provider.chat(request, effective_config=effective_config)
            except BaseException as exc:
                attempt += 1
                if not self.retry_policy.should_retry(exc, attempt):
                    raise
                await asyncio.sleep(self.retry_policy.backoff_for_attempt(attempt, exc))

    async def _aggregate_stream_response(
        self,
        request: ChatRequest,
        effective_config: BaseLLMConfig,
    ) -> ChatResponse:
        content_parts: list[str] = []
        final_tool_calls = None
        finish_reason: str | None = None
        usage = None
        model: str | None = None
        raw_response: dict[str, Any] | None = None

        async for chunk in self.chat_stream(
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            config_overrides=request.config_overrides,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.completed_tool_calls is not None:
                final_tool_calls = chunk.completed_tool_calls
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.raw_chunk is not None:
                raw_response = chunk.raw_chunk
                chunk_model = chunk.raw_chunk.get("model")
                if isinstance(chunk_model, str):
                    model = chunk_model

        content = "".join(content_parts) if content_parts else None
        return ChatResponse(
            content=content,
            tool_calls=final_tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw_response,
            model=model,
            provider=self._provider.provider_name,
        )

    def _apply_overrides(self, config_overrides: dict[str, Any] | None) -> BaseLLMConfig:
        if not config_overrides:
            return self.config

        if not isinstance(config_overrides, dict):
            raise ConfigValidationError("config_overrides must be a dictionary")

        data = self.config.model_dump(by_alias=True)
        merged = deep_merge(data, config_overrides)

        config_cls = type(self.config)
        try:
            validated = config_cls.model_validate(merged)
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc
        return validated

    def _create_provider(self, config: BaseLLMConfig) -> BaseProvider:
        spec = _provider_spec_for_config(config)
        return spec.provider_type(config, timeout=self.timeout)


class SyncConduit:
    """Synchronous wrapper around the async Conduit client."""

    def __init__(
        self,
        config: BaseLLMConfig,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._async_client = Conduit(
            config=config,
            retry_policy=retry_policy,
            timeout=timeout,
        )
        self._loop = asyncio.new_event_loop()
        self._closed = False

    def __enter__(self) -> "SyncConduit":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self._ensure_open()
        return self._run(
            self._async_client.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                config_overrides=config_overrides,
            )
        )

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> Iterator[ChatResponseChunk]:
        self._ensure_open()
        self._ensure_not_running_loop()

        async_iter = self._async_client.chat_stream(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            config_overrides=config_overrides,
        )
        iterator = async_iter.__aiter__()

        try:
            while True:
                try:
                    chunk = self._run(iterator.__anext__())
                except StopAsyncIteration:
                    break
                yield chunk
        finally:
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                try:
                    self._run(aclose())
                except RuntimeError:
                    pass

    def close(self) -> None:
        if self._closed:
            return

        self._ensure_not_running_loop()
        try:
            self._loop.run_until_complete(self._async_client.aclose())
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
        finally:
            self._closed = True
            if not self._loop.is_closed():
                self._loop.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SyncConduit client is closed")

    @staticmethod
    def _ensure_not_running_loop() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            "SyncConduit cannot run inside an existing event loop; use Conduit instead"
        )

    def _run(self, awaitable: Awaitable[T]) -> T:
        self._ensure_open()
        self._ensure_not_running_loop()
        return self._loop.run_until_complete(awaitable)


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries with scalar/list replacement semantics."""
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
