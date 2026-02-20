from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from conduit.config.vllm import VLLMConfig
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


class VLLMProvider(BaseProvider[VLLMConfig]):
    """vLLM provider using the OpenAI-compatible chat completions API."""

    provider_name = "vllm"

    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: VLLMConfig,
        stream: bool,
    ) -> dict[str, Any]:
        ensure_tool_strict_supported(
            request.tools,
            provider_name=self.provider_name,
            supports_tool_strict=False,
        )

        body: dict[str, Any] = {
            "model": effective_config.model,
            "messages": to_openai_messages(request.messages),
            "stream": stream,
            "temperature": effective_config.temperature,
            "max_tokens": effective_config.max_tokens,
            "max_completion_tokens": effective_config.max_completion_tokens,
            "response_format": effective_config.response_format,
            "top_p": effective_config.top_p,
            "stop": normalize_stop(effective_config.stop),
            "seed": effective_config.seed,
            "frequency_penalty": effective_config.frequency_penalty,
            "presence_penalty": effective_config.presence_penalty,
            "n": effective_config.n,
            "logprobs": effective_config.logprobs,
            "top_logprobs": effective_config.top_logprobs,
            "tools": tool_definitions_to_openai(request.tools),
            "tool_choice": tool_choice_to_openai_payload(request.tool_choice),
            "parallel_tool_calls": effective_config.parallel_tool_calls,
            "use_beam_search": effective_config.use_beam_search,
            "top_k": effective_config.top_k,
            "min_p": effective_config.min_p,
            "repetition_penalty": effective_config.repetition_penalty,
            "length_penalty": effective_config.length_penalty,
            "stop_token_ids": effective_config.stop_token_ids,
            "include_stop_str_in_output": effective_config.include_stop_str_in_output,
            "ignore_eos": effective_config.ignore_eos,
            "min_tokens": effective_config.min_tokens,
            "truncate_prompt_tokens": effective_config.truncate_prompt_tokens,
            "prompt_logprobs": effective_config.prompt_logprobs,
            "echo": effective_config.echo,
            "add_generation_prompt": effective_config.add_generation_prompt,
            "continue_final_message": effective_config.continue_final_message,
            "chat_template": effective_config.chat_template,
            "chat_template_kwargs": effective_config.chat_template_kwargs,
        }

        if effective_config.stream_options is not None:
            if not stream:
                raise ConfigValidationError(
                    "vLLM stream_options is only supported when stream=True"
                )
            body["stream_options"] = effective_config.stream_options

        if effective_config.structured_outputs is not None:
            body["structured_outputs"] = effective_config.structured_outputs.model_dump(
                exclude_none=True,
                by_alias=True,
            )

        return drop_nones(body)

    async def chat(
        self,
        request: ChatRequest,
        *,
        effective_config: VLLMConfig,
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
            raise ResponseParseError("vLLM response did not contain choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ResponseParseError("vLLM response choice has unexpected shape")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ResponseParseError("vLLM response choice did not contain a message")

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
        effective_config: VLLMConfig,
    ) -> AsyncIterator[ChatResponseChunk]:
        payload = self.build_request_body(
            request,
            effective_config=effective_config,
            stream=True,
        )
        headers = self.default_headers(effective_config=effective_config)
        url = self.make_url("/chat/completions", effective_config=effective_config)

        # Tool-call completion state machine:
        # - accumulator: reassembles streamed tool-call fragments into complete calls
        # - saw_tool_call_delta: tracks whether any tool-call delta appeared in the stream
        # - allow_terminal_tool_call_flush: gate that disables the terminal flush when
        #   should_complete_tool_calls() returns False (e.g. finish_reason="length")
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
                        raw_chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise StreamError(f"invalid SSE JSON payload: {exc}") from exc

                    if not isinstance(raw_chunk, dict):
                        raise StreamError("stream chunk must decode to an object")

                    usage = parse_usage(raw_chunk.get("usage"))
                    choices = raw_chunk.get("choices")
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
                            raw_chunk=raw_chunk,
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
                            raw_chunk=raw_chunk,
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

                    completed_tool_calls = None
                    if finish_reason is not None:
                        if should_complete_tool_calls(
                            finish_reason=finish_reason,
                            saw_tool_call_delta=saw_tool_call_delta,
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
                        raw_chunk=raw_chunk,
                    )

                    if not should_emit_stream_chunk(chunk):
                        continue
                    yield chunk

            if allow_terminal_tool_call_flush and not emitted_completed_tool_calls:
                completed_tool_calls = accumulator.completed_calls() or None
                if completed_tool_calls is not None:
                    yield ChatResponseChunk(completed_tool_calls=completed_tool_calls)
        except httpx.HTTPError as exc:
            raise StreamError(f"vLLM streaming request failed: {exc}") from exc
