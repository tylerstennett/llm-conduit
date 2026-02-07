from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

import httpx

from conduit.config.base import BaseLLMConfig
from conduit.config.ollama import OllamaConfig
from conduit.exceptions import (
    ConfigValidationError,
    ResponseParseError,
    StreamError,
    ToolCallParseError,
)
from conduit.models.messages import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    Message,
    PartialToolCall,
    Role,
    ToolCallChunkAccumulator,
    UsageStats,
)
from conduit.providers.base import BaseProvider
from conduit.providers.streaming import iter_ndjson
from conduit.providers.utils import drop_nones
from conduit.tools.schema import ToolCall, parse_tool_arguments


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
        stream: bool,
    ) -> dict[str, Any]:
        config = cast(OllamaConfig, effective_config)

        include_tools = False
        if request.tools:
            if request.tool_choice is None or request.tool_choice == "auto":
                include_tools = True
            elif request.tool_choice == "none":
                include_tools = False
            else:
                raise ConfigValidationError(
                    "Ollama only supports tool_choice values of auto or none"
                )

        id_to_name = build_tool_id_to_name_map(request.messages)
        messages = to_ollama_messages(request.messages, id_to_name=id_to_name)

        options = self._build_options(config)

        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": stream,
            "options": options if options else None,
            "format": config.format,
            "keep_alive": config.keep_alive,
            "think": config.think,
            "logprobs": config.logprobs,
            "top_logprobs": config.top_logprobs,
        }
        if include_tools:
            body["tools"] = [tool.to_openai_tool() for tool in request.tools or []]

        return drop_nones(body)

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
            "/api/chat",
            payload,
            effective_config=effective_config,
        )

        message = raw.get("message")
        if not isinstance(message, dict):
            raise ResponseParseError("Ollama response did not contain message object")

        return ChatResponse(
            content=message.get("content") if isinstance(message.get("content"), str) else None,
            tool_calls=parse_ollama_tool_calls(message.get("tool_calls")),
            finish_reason=raw.get("done_reason") if isinstance(raw.get("done_reason"), str) else None,
            usage=parse_ollama_usage(raw),
            raw_response=raw,
            model=raw.get("model") if isinstance(raw.get("model"), str) else None,
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
        url = self.make_url("/api/chat", effective_config=effective_config)
        accumulator = ToolCallChunkAccumulator()

        try:
            async with self.http_client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as response:
                await self.raise_for_stream_status(response)

                async for raw_chunk in iter_ndjson(response):
                    message = raw_chunk.get("message")
                    if not isinstance(message, dict):
                        message = {}

                    content = message.get("content")
                    if not isinstance(content, str):
                        content = None

                    partial_calls = parse_ollama_partial_tool_calls(message.get("tool_calls"))
                    if partial_calls:
                        for partial in partial_calls:
                            accumulator.ingest(partial)

                    done = raw_chunk.get("done")
                    finish_reason = (
                        raw_chunk.get("done_reason")
                        if done and isinstance(raw_chunk.get("done_reason"), str)
                        else None
                    )

                    completed_tool_calls = None
                    if done:
                        completed_tool_calls = accumulator.completed_calls() or None

                    usage = parse_ollama_usage(raw_chunk) if done else None

                    chunk = ChatResponseChunk(
                        content=content,
                        tool_calls=partial_calls,
                        completed_tool_calls=completed_tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw_chunk=raw_chunk,
                    )

                    if (
                        chunk.content is None
                        and chunk.tool_calls is None
                        and chunk.completed_tool_calls is None
                        and chunk.finish_reason is None
                        and chunk.usage is None
                    ):
                        continue
                    yield chunk
        except httpx.HTTPError as exc:
            raise StreamError(f"Ollama streaming request failed: {exc}") from exc

    @staticmethod
    def _build_options(config: OllamaConfig) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "seed": config.seed,
            "stop": config.stop,
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict if config.num_predict is not None else config.max_tokens,
            "num_gpu": config.num_gpu,
            "num_thread": config.num_thread,
            "num_keep": config.num_keep,
            "num_batch": config.num_batch,
            "repeat_penalty": config.repeat_penalty,
            "repeat_last_n": config.repeat_last_n,
            "tfs_z": config.tfs_z,
            "typical_p": config.typical_p,
            "mirostat": config.mirostat,
            "mirostat_tau": config.mirostat_tau,
            "mirostat_eta": config.mirostat_eta,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
        return drop_nones(options)


def build_tool_id_to_name_map(messages: list[Message]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for message in messages:
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            continue
        for index, tool_call in enumerate(message.tool_calls):
            mapping[tool_call.id] = tool_call.name
            fallback_id = f"ollama_call_{index}"
            mapping.setdefault(fallback_id, tool_call.name)
    return mapping


def to_ollama_messages(
    messages: list[Message],
    *,
    id_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for message in messages:
        payload: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content or "",
        }

        if message.images:
            payload["images"] = message.images

        if message.role is Role.ASSISTANT and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "index": index,
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for index, call in enumerate(message.tool_calls)
            ]

        if message.role is Role.TOOL:
            if not message.tool_call_id:
                raise ToolCallParseError(
                    "tool messages require tool_call_id for Ollama conversion"
                )
            tool_name = id_to_name.get(message.tool_call_id)
            if tool_name is None:
                raise ToolCallParseError(
                    "could not resolve tool_call_id to tool_name for Ollama request"
                )
            payload["tool_name"] = tool_name

        output.append(payload)

    return output


def parse_ollama_tool_calls(raw_tool_calls: Any) -> list[ToolCall] | None:
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return None

    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = parse_tool_arguments(function.get("arguments"))
        parsed.append(
            ToolCall(
                id=f"ollama_call_{index}",
                name=name,
                arguments=arguments,
            )
        )
    return parsed or None


def parse_ollama_partial_tool_calls(raw_tool_calls: Any) -> list[PartialToolCall] | None:
    if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
        return None

    partials: list[PartialToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            name = None

        arguments = function.get("arguments")
        arguments_fragment: str | None = None
        if isinstance(arguments, str):
            arguments_fragment = arguments
        elif isinstance(arguments, dict):
            arguments_fragment = json.dumps(arguments)

        partials.append(
            PartialToolCall(
                index=index,
                id=f"ollama_call_{index}",
                name=name,
                arguments_fragment=arguments_fragment,
            )
        )

    return partials or None


def parse_ollama_usage(raw: dict[str, Any]) -> UsageStats | None:
    prompt_eval_count = raw.get("prompt_eval_count")
    eval_count = raw.get("eval_count")

    prompt_tokens = prompt_eval_count if isinstance(prompt_eval_count, int) else None
    completion_tokens = eval_count if isinstance(eval_count, int) else None
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return UsageStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
