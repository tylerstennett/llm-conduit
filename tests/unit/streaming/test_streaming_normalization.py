from __future__ import annotations

import httpx
import pytest

from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.models.messages import ChatRequest
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openrouter import OpenRouterProvider
from conduit.providers.vllm import VLLMProvider


@pytest.mark.asyncio
async def test_vllm_sse_stream_assembles_content(sample_messages) -> None:
    sse_payload = (
        'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}'
        "\n\n"
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
        "\n\n"
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
    )
    client = httpx.AsyncClient(transport=transport)
    provider = VLLMProvider(VLLMConfig(model="m"), http_client=client)

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            ChatRequest(messages=sample_messages, stream=True),
            effective_config=provider.config,
        )
    ]

    assert "".join(chunk.content or "" for chunk in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 2

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_assembles_tool_calls(sample_messages, sample_tools) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{"}}]},"finish_reason":null}]}}'
        "\n\n"
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"\\"location\\":\\"NYC\\"}"}}]},"finish_reason":"tool_calls"}]}}'
        "\n\n"
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OpenRouterProvider(
        OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"),
        http_client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto", stream=True),
            effective_config=provider.config,
        )
    ]

    completed = [chunk for chunk in chunks if chunk.completed_tool_calls]
    assert completed
    final_calls = completed[-1].completed_tool_calls
    assert final_calls is not None
    assert final_calls[0].id == "call_1"
    assert final_calls[0].name == "get_weather"
    assert final_calls[0].arguments["location"] == "NYC"

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_emits_usage_only_terminal_chunk(sample_messages) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}}'
        "\n\n"
        'data: {"data":{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}'
        "\n\n"
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OpenRouterProvider(
        OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"),
        http_client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            ChatRequest(messages=sample_messages, stream=True),
            effective_config=provider.config,
        )
    ]

    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert usage_chunks
    assert usage_chunks[-1].usage is not None
    assert usage_chunks[-1].usage.total_tokens == 2

    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_ndjson_stream_normalizes_chunks(sample_messages, sample_tools) -> None:
    ndjson_payload = "\n".join(
        [
            '{"model":"qwen3","message":{"role":"assistant","content":"Hi"},"done":false}',
            '{"model":"qwen3","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"NYC"}}}]},"done":true,"done_reason":"tool_calls","prompt_eval_count":5,"eval_count":2}',
        ]
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=ndjson_payload, headers={"content-type": "application/x-ndjson"})
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(OllamaConfig(model="qwen3"), http_client=client)

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            ChatRequest(messages=sample_messages, tools=sample_tools, stream=True),
            effective_config=provider.config,
        )
    ]

    assert chunks[0].content == "Hi"
    assert chunks[-1].completed_tool_calls is not None
    assert chunks[-1].completed_tool_calls[0].name == "get_weather"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 7

    await client.aclose()
