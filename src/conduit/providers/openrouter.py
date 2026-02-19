from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

import httpx

from conduit.config.base import BaseLLMConfig
from conduit.config.openrouter import OpenRouterConfig
from conduit.exceptions import ConfigValidationError, ResponseParseError, StreamError
from conduit.models.messages import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
)
from conduit.providers.base import BaseProvider
from conduit.providers.openai_format import (
    ensure_tool_strict_supported,
    extract_openai_message_content,
    normalize_stop,
    parse_openai_tool_calls,
    parse_usage,
    to_openai_messages,
    tool_choice_to_openai_payload,
    tool_definitions_to_openai,
)
from conduit.providers.streaming import (
    ToolCallChunkAccumulator,
    iter_sse_data,
    parse_openai_stream_tool_calls,
)
from conduit.providers.utils import drop_nones
from conduit.utils.streaming import should_complete_tool_calls, should_emit_stream_chunk


class OpenRouterProvider(BaseProvider):
    provider_name = "openrouter"
    supported_runtime_override_keys = frozenset({"openrouter_context_metadata_fields"})

    def default_headers(
        self,
        *,
        effective_config: BaseLLMConfig | None = None,
    ) -> dict[str, str]:
        headers = super().default_headers(effective_config=effective_config)
        active_config = effective_config if effective_config is not None else self.config
        config = cast(OpenRouterConfig, active_config)
        if config.app_url:
            headers["HTTP-Referer"] = config.app_url
        if config.app_name:
            headers["X-Title"] = config.app_name
        return headers

    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
        stream: bool,
    ) -> dict[str, Any]:
        config = cast(OpenRouterConfig, effective_config)
        ensure_tool_strict_supported(
            request.tools,
            provider_name=self.provider_name,
            supports_tool_strict=True,
        )

        body: dict[str, Any] = {
            "model": config.model,
            "models": config.models,
            "messages": to_openai_messages(request.messages),
            "stream": stream,
            "stream_options": config.stream_options,
            "reasoning": config.reasoning,
            "transforms": config.transforms,
            "include": config.include,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_completion_tokens": config.max_completion_tokens,
            "top_p": config.top_p,
            "stop": normalize_stop(config.stop),
            "seed": config.seed,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "n": config.n,
            "logprobs": config.logprobs,
            "top_logprobs": config.top_logprobs,
            "logit_bias": config.logit_bias,
            "response_format": config.response_format,
            "provider": (
                config.provider_prefs.model_dump(exclude_none=True)
                if config.provider_prefs is not None
                else None
            ),
            "route": config.route,
            "metadata": self._resolve_metadata(
                config_metadata=config.metadata,
                request=request,
            ),
            "plugins": config.plugins,
            "user": config.user,
            "tools": tool_definitions_to_openai(
                request.tools,
                include_strict=True,
            ),
            "tool_choice": tool_choice_to_openai_payload(request.tool_choice),
        }

        return drop_nones(body)

    def _resolve_metadata(
        self,
        *,
        config_metadata: dict[str, str] | None,
        request: ChatRequest,
    ) -> dict[str, str] | None:
        selected_fields = self._context_metadata_fields(request.runtime_overrides)
        if not selected_fields:
            return config_metadata

        metadata: dict[str, str] = dict(config_metadata or {})
        context = request.context
        if context is None:
            return metadata or None

        if "thread_id" in selected_fields and context.thread_id is not None:
            metadata["conduit_context_thread_id"] = context.thread_id
        if "tags" in selected_fields:
            metadata["conduit_context_tags"] = json.dumps(
                context.tags,
                separators=(",", ":"),
            )
        if "metadata" in selected_fields:
            metadata["conduit_context_metadata"] = json.dumps(
                context.metadata,
                separators=(",", ":"),
            )
        return metadata or None

    @staticmethod
    def _context_metadata_fields(
        runtime_overrides: dict[str, Any] | None,
    ) -> set[str]:
        if not runtime_overrides:
            return set()

        raw_fields = runtime_overrides.get("openrouter_context_metadata_fields")
        if raw_fields is None:
            return set()
        if not isinstance(raw_fields, list):
            raise ConfigValidationError(
                "openrouter_context_metadata_fields must be a list of strings"
            )

        allowed_fields = {"thread_id", "tags", "metadata"}
        parsed: set[str] = set()
        for raw_field in raw_fields:
            if not isinstance(raw_field, str):
                raise ConfigValidationError(
                    "openrouter_context_metadata_fields must contain only strings"
                )
            if raw_field not in allowed_fields:
                raise ConfigValidationError(
                    "openrouter_context_metadata_fields entries must be one of: "
                    "thread_id, tags, metadata"
                )
            parsed.add(raw_field)
        return parsed

    async def chat(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
    ) -> ChatResponse:
        payload = self.build_request_body(
            request,
            effective_config=effective_config,
            stream=False,
        )
        raw = await self.post_json(
            "/chat/completions",
            payload,
            effective_config=effective_config,
        )

        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ResponseParseError("OpenRouter response did not contain choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ResponseParseError("OpenRouter response choice has unexpected shape")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ResponseParseError("OpenRouter response choice did not contain a message")

        return ChatResponse(
            content=extract_openai_message_content(message.get("content")),
            tool_calls=parse_openai_tool_calls(message.get("tool_calls")),
            finish_reason=first_choice.get("finish_reason"),
            usage=parse_usage(raw.get("usage")),
            raw_response=raw,
            model=raw.get("model"),
            provider=self.provider_name,
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
    ) -> AsyncIterator[ChatResponseChunk]:
        payload = self.build_request_body(
            request,
            effective_config=effective_config,
            stream=True,
        )
        headers = self.default_headers(effective_config=effective_config)
        url = self.make_url("/chat/completions", effective_config=effective_config)
        accumulator = ToolCallChunkAccumulator()
        emitted_completed_tool_calls = False
        saw_tool_call_delta = False
        allow_terminal_tool_call_flush = True

        try:
            async with self.http_client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as response:
                await self.raise_for_stream_status(response)

                async for data in iter_sse_data(response):
                    if data == "[DONE]":
                        if (
                            allow_terminal_tool_call_flush
                            and not emitted_completed_tool_calls
                        ):
                            completed_tool_calls = accumulator.completed_calls() or None
                            if completed_tool_calls is not None:
                                emitted_completed_tool_calls = True
                                yield ChatResponseChunk(completed_tool_calls=completed_tool_calls)
                        break

                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise StreamError(f"invalid OpenRouter SSE JSON payload: {exc}") from exc

                    if not isinstance(parsed, dict):
                        raise StreamError("OpenRouter stream payload must be a JSON object")

                    raw_chunk = parsed.get("data")
                    if isinstance(raw_chunk, dict):
                        chunk_data = raw_chunk
                    else:
                        chunk_data = parsed

                    usage = parse_usage(chunk_data.get("usage"))

                    choices = chunk_data.get("choices")
                    if not isinstance(choices, list) or not choices:
                        completed_tool_calls = None
                        if (
                            usage is not None
                            and allow_terminal_tool_call_flush
                            and not emitted_completed_tool_calls
                        ):
                            completed_tool_calls = accumulator.completed_calls() or None
                            if completed_tool_calls is not None:
                                emitted_completed_tool_calls = True
                        chunk = ChatResponseChunk(
                            completed_tool_calls=completed_tool_calls,
                            usage=usage,
                            raw_chunk=chunk_data,
                        )
                        if not should_emit_stream_chunk(chunk):
                            continue
                        yield chunk
                        continue

                    first_choice = choices[0]
                    if not isinstance(first_choice, dict):
                        completed_tool_calls = None
                        if (
                            usage is not None
                            and allow_terminal_tool_call_flush
                            and not emitted_completed_tool_calls
                        ):
                            completed_tool_calls = accumulator.completed_calls() or None
                            if completed_tool_calls is not None:
                                emitted_completed_tool_calls = True
                        chunk = ChatResponseChunk(
                            completed_tool_calls=completed_tool_calls,
                            usage=usage,
                            raw_chunk=chunk_data,
                        )
                        if not should_emit_stream_chunk(chunk):
                            continue
                        yield chunk
                        continue

                    delta = first_choice.get("delta")
                    if not isinstance(delta, dict):
                        delta = {}

                    content = delta.get("content")
                    if not isinstance(content, str):
                        content = None

                    partial_calls = parse_openai_stream_tool_calls(delta.get("tool_calls"))
                    if partial_calls:
                        saw_tool_call_delta = True
                        for partial in partial_calls:
                            accumulator.ingest(partial)

                    raw_finish_reason = first_choice.get("finish_reason")
                    finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None
                    raw_native_finish_reason = first_choice.get("native_finish_reason")
                    native_finish_reason = (
                        raw_native_finish_reason
                        if isinstance(raw_native_finish_reason, str)
                        else None
                    )

                    completed_tool_calls = None
                    if finish_reason is not None or native_finish_reason is not None:
                        if should_complete_tool_calls(
                            finish_reason=finish_reason,
                            saw_tool_call_delta=saw_tool_call_delta,
                            native_finish_reason=native_finish_reason,
                        ):
                            if not emitted_completed_tool_calls:
                                completed_tool_calls = accumulator.completed_calls() or None
                                if completed_tool_calls is not None:
                                    emitted_completed_tool_calls = True
                        else:
                            allow_terminal_tool_call_flush = False

                    chunk = ChatResponseChunk(
                        content=content,
                        tool_calls=partial_calls,
                        completed_tool_calls=completed_tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw_chunk=chunk_data,
                    )

                    if not should_emit_stream_chunk(chunk):
                        continue
                    yield chunk

            if allow_terminal_tool_call_flush and not emitted_completed_tool_calls:
                completed_tool_calls = accumulator.completed_calls() or None
                if completed_tool_calls is not None:
                    yield ChatResponseChunk(completed_tool_calls=completed_tool_calls)
        except httpx.HTTPError as exc:
            raise StreamError(f"OpenRouter streaming request failed: {exc}") from exc
