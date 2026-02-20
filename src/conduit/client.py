from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, TypeVar

from pydantic import ValidationError

from conduit.config import BaseLLMConfig, OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.models.messages import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    Message,
    RequestContext,
    StreamEvent,
    StreamEventAccumulator,
)
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
    """Provider-aware asynchronous chat client.

    Args:
        config: Provider-specific configuration (e.g. ``VLLMConfig``).
        retry_policy: Optional retry policy for transient failures.
        timeout: HTTP request timeout in seconds.
        strict_runtime_overrides: When ``True``, raise on unrecognised
            runtime override keys instead of silently dropping them.
    """

    def __init__(
        self,
        config: BaseLLMConfig,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
        strict_runtime_overrides: bool = False,
    ) -> None:
        self.config = config
        self.retry_policy = retry_policy
        self.timeout = timeout
        self.strict_runtime_overrides = strict_runtime_overrides
        self._provider = self._create_provider(config)

    async def __aenter__(self) -> "Conduit":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._provider.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        config_overrides: dict[str, Any] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the full response.

        When *stream* is ``True`` the response is aggregated from a streaming
        call via ``StreamEventAccumulator``.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy (``"auto"``, ``"none"``,
                ``"required"``, or a specific tool dict).
            stream: If ``True``, aggregate streaming chunks into a single
                ``ChatResponse``.
            config_overrides: Per-request config overrides merged onto the
                base config.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Returns:
            The completed chat response.
        """
        normalized_runtime_overrides = self._normalize_runtime_overrides(
            runtime_overrides
        )
        request = ChatRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            context=context,
            runtime_overrides=normalized_runtime_overrides,
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
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatResponseChunk]:
        """Stream chat completion chunks as they arrive.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy.
            config_overrides: Per-request config overrides.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Yields:
            ChatResponseChunk: Individual stream chunks.
        """
        normalized_runtime_overrides = self._normalize_runtime_overrides(
            runtime_overrides
        )
        request = ChatRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            context=context,
            runtime_overrides=normalized_runtime_overrides,
        )
        effective_config = self._apply_overrides(config_overrides)

        async for chunk in self._chat_stream_with_retry(request, effective_config):
            yield chunk

    async def chat_events(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream high-level typed events (text deltas, tool calls, etc.).

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy.
            config_overrides: Per-request config overrides.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Yields:
            StreamEvent: Typed events such as ``text_delta``, ``tool_call_delta``,
                ``tool_call_completed``, ``usage``, ``finish``, or ``error``.
        """
        normalized_runtime_overrides = self._normalize_runtime_overrides(
            runtime_overrides
        )
        request = ChatRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            context=context,
            runtime_overrides=normalized_runtime_overrides,
        )
        effective_config = self._apply_overrides(config_overrides)

        async for event in self._events_from_chunk_stream(
            self._chat_stream_with_retry(request, effective_config)
        ):
            yield event

    @staticmethod
    def from_env(provider: str = "openrouter") -> "Conduit":
        """Create a Conduit client from environment variables.

        Args:
            provider: Provider name (``"vllm"``, ``"ollama"``, or
                ``"openrouter"``).

        Returns:
            A configured ``Conduit`` instance.

        Raises:
            ConfigValidationError: If required environment variables are
                missing or the provider name is invalid.
        """
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

    async def _chat_stream_with_retry(
        self,
        request: ChatRequest,
        effective_config: BaseLLMConfig,
    ) -> AsyncIterator[ChatResponseChunk]:
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

    async def _aggregate_stream_response(
        self,
        request: ChatRequest,
        effective_config: BaseLLMConfig,
    ) -> ChatResponse:
        accumulator = StreamEventAccumulator()
        async for event in self._events_from_chunk_stream(
            self._chat_stream_with_retry(request, effective_config)
        ):
            accumulator.ingest(event)
        return accumulator.to_response(provider=self._provider.provider_name)

    async def _events_from_chunk_stream(
        self,
        chunks: AsyncIterator[ChatResponseChunk],
    ) -> AsyncIterator[StreamEvent]:
        try:
            async for chunk in chunks:
                if chunk.content:
                    yield StreamEvent(
                        type="text_delta",
                        text=chunk.content,
                        raw=chunk.raw_chunk,
                    )

                if chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        yield StreamEvent(
                            type="tool_call_delta",
                            tool_call=tool_call,
                            raw=chunk.raw_chunk,
                        )

                if chunk.completed_tool_calls:
                    for tool_call in chunk.completed_tool_calls:
                        yield StreamEvent(
                            type="tool_call_completed",
                            tool_call=tool_call,
                            raw=chunk.raw_chunk,
                        )

                if chunk.usage is not None:
                    yield StreamEvent(
                        type="usage",
                        usage=chunk.usage,
                        raw=chunk.raw_chunk,
                    )

                if chunk.finish_reason is not None:
                    yield StreamEvent(
                        type="finish",
                        finish_reason=chunk.finish_reason,
                        raw=chunk.raw_chunk,
                    )
        except Exception as exc:
            yield StreamEvent(
                type="error",
                error=str(exc),
                raw={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise

    def _normalize_runtime_overrides(
        self,
        runtime_overrides: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if runtime_overrides is None:
            return None
        if not isinstance(runtime_overrides, dict):
            raise ConfigValidationError("runtime_overrides must be a dictionary")

        supported_keys = self._provider.supported_runtime_override_keys
        unknown_keys = [key for key in runtime_overrides if key not in supported_keys]
        if unknown_keys and self.strict_runtime_overrides:
            unknown_keys_text = sorted(str(key) for key in unknown_keys)
            raise ConfigValidationError(
                "unknown runtime_overrides keys for "
                f"{self._provider.provider_name}: {', '.join(unknown_keys_text)}"
            )

        return {
            key: value
            for key, value in runtime_overrides.items()
            if key in supported_keys
        }

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
    """Synchronous wrapper around the async ``Conduit`` client.

    Bridges async to sync via a dedicated event loop. Must not be used
    inside an already-running event loop; use ``Conduit`` directly instead.

    Args:
        config: Provider-specific configuration (e.g. ``VLLMConfig``).
        retry_policy: Optional retry policy for transient failures.
        timeout: HTTP request timeout in seconds.
        strict_runtime_overrides: When ``True``, raise on unrecognised
            runtime override keys instead of silently dropping them.
    """

    def __init__(
        self,
        config: BaseLLMConfig,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
        strict_runtime_overrides: bool = False,
    ) -> None:
        self._async_client = Conduit(
            config=config,
            retry_policy=retry_policy,
            timeout=timeout,
            strict_runtime_overrides=strict_runtime_overrides,
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
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the full response.

        Synchronous equivalent of :meth:`Conduit.chat`.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy.
            stream: If ``True``, aggregate streaming chunks into a single
                ``ChatResponse``.
            config_overrides: Per-request config overrides.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Returns:
            The completed chat response.
        """
        self._ensure_open()
        return self._run(
            self._async_client.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                config_overrides=config_overrides,
                context=context,
                runtime_overrides=runtime_overrides,
            )
        )

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> Iterator[ChatResponseChunk]:
        """Stream chat completion chunks as they arrive.

        Synchronous equivalent of :meth:`Conduit.chat_stream`.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy.
            config_overrides: Per-request config overrides.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Yields:
            ChatResponseChunk: Individual stream chunks.
        """
        yield from self._iter_async(
            self._async_client.chat_stream(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                config_overrides=config_overrides,
                context=context,
                runtime_overrides=runtime_overrides,
            )
        )

    def chat_events(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        config_overrides: dict[str, Any] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream high-level typed events.

        Synchronous equivalent of :meth:`Conduit.chat_events`.

        Args:
            messages: Conversation messages.
            tools: Tool definitions available to the model.
            tool_choice: Tool selection strategy.
            config_overrides: Per-request config overrides.
            context: Optional run-scoped metadata.
            runtime_overrides: Provider-specific runtime overrides.

        Yields:
            StreamEvent: Typed events such as ``text_delta``, ``tool_call_delta``,
                ``tool_call_completed``, ``usage``, ``finish``, or ``error``.
        """
        yield from self._iter_async(
            self._async_client.chat_events(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                config_overrides=config_overrides,
                context=context,
                runtime_overrides=runtime_overrides,
            )
        )

    def _iter_async(self, async_iter: AsyncIterator[T]) -> Iterator[T]:
        self._ensure_open()
        self._ensure_not_running_loop()
        iterator = async_iter.__aiter__()
        try:
            while True:
                try:
                    item = self._run(iterator.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        finally:
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                try:
                    self._run(aclose())
                except RuntimeError:
                    pass

    def close(self) -> None:
        """Close the underlying event loop and HTTP client."""
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
    """Deep-merge two dictionaries with scalar/list replacement semantics.

    Nested dicts are merged recursively; all other values (including lists)
    in *updates* replace the corresponding value in *base*.

    Args:
        base: The base dictionary.
        updates: Values to merge on top of *base*.

    Returns:
        A new merged dictionary.
    """
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged
