from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

import httpx

from conduit.config.base import BaseLLMConfig
from conduit.config.vllm import VLLMConfig
from conduit.exceptions import StreamError
from conduit.models.messages import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    ToolCallChunkAccumulator,
)
from conduit.providers.base import (
    BaseProvider,
    normalize_stop,
    parse_openai_tool_calls,
    parse_usage,
    to_openai_messages,
    tool_choice_to_openai_payload,
    tool_definitions_to_openai,
)
from conduit.providers.streaming import iter_sse_data, parse_openai_stream_tool_calls


class VLLMProvider(BaseProvider):
    provider_name = "vllm"

    def build_request_body(
        self,
        request: ChatRequest,
        *,
        effective_config: BaseLLMConfig,
        stream: bool,
    ) -> dict[str, Any]:
        config = cast(VLLMConfig, effective_config)

        body: dict[str, Any] = {
            "model": config.model,
            "messages": to_openai_messages(request.messages),
            "stream": stream,
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
            "tools": tool_definitions_to_openai(request.tools),
            "tool_choice": tool_choice_to_openai_payload(request.tool_choice),
            "parallel_tool_calls": config.parallel_tool_calls,
            "use_beam_search": config.use_beam_search,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "repetition_penalty": config.repetition_penalty,
            "length_penalty": config.length_penalty,
            "stop_token_ids": config.stop_token_ids,
            "include_stop_str_in_output": config.include_stop_str_in_output,
            "ignore_eos": config.ignore_eos,
            "min_tokens": config.min_tokens,
            "truncate_prompt_tokens": config.truncate_prompt_tokens,
            "prompt_logprobs": config.prompt_logprobs,
            "echo": config.echo,
            "add_generation_prompt": config.add_generation_prompt,
            "continue_final_message": config.continue_final_message,
            "chat_template": config.chat_template,
            "chat_template_kwargs": config.chat_template_kwargs,
        }

        structured_outputs = config.merged_structured_outputs()
        if structured_outputs is not None:
            body["structured_outputs"] = structured_outputs.model_dump(
                exclude_none=True,
                by_alias=True,
            )

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
            "/chat/completions",
            payload,
            effective_config=effective_config,
        )

        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise StreamError("vLLM response did not contain choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise StreamError("vLLM response choice has unexpected shape")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise StreamError("vLLM response choice did not contain a message")

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

        try:
            async with self.http_client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    raise self.map_http_error(response)

                async for data in iter_sse_data(response):
                    if data == "[DONE]":
                        break

                    try:
                        raw_chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise StreamError(f"invalid SSE JSON payload: {exc}") from exc

                    if not isinstance(raw_chunk, dict):
                        raise StreamError("stream chunk must decode to an object")

                    choices = raw_chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue

                    first_choice = choices[0]
                    if not isinstance(first_choice, dict):
                        continue

                    delta = first_choice.get("delta")
                    if not isinstance(delta, dict):
                        delta = {}

                    content = delta.get("content")
                    if not isinstance(content, str):
                        content = None

                    partial_calls = parse_openai_stream_tool_calls(delta.get("tool_calls"))
                    if partial_calls:
                        for partial in partial_calls:
                            accumulator.ingest(partial)

                    finish_reason = first_choice.get("finish_reason")
                    if finish_reason is not None and not isinstance(finish_reason, str):
                        finish_reason = None

                    usage = parse_usage(raw_chunk.get("usage"))

                    completed_tool_calls = None
                    if finish_reason == "tool_calls":
                        completed_tool_calls = accumulator.completed_calls() or None

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
            raise StreamError(f"vLLM streaming request failed: {exc}") from exc


def extract_openai_message_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None


def drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: drop_nones(inner)
            for key, inner in value.items()
            if inner is not None
        }
    if isinstance(value, list):
        return [drop_nones(item) for item in value]
    return value
