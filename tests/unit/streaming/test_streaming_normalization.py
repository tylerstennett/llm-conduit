from __future__ import annotations

import httpx
import pytest

from conduit.client import Conduit
from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.models.messages import ChatRequest
from conduit.tools.schema import ToolCall
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
    assert chunks[-1].usage.prompt_tokens == 1
    assert chunks[-1].usage.completion_tokens == 1
    assert chunks[-1].usage.total_tokens == 2

    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_sse_stream_assembles_tool_calls_with_stop_finish_reason(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{"}}]},"finish_reason":null}]}'
        "\n\n"
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"\\"location\\":\\"NYC\\"}"}}]},"finish_reason":"stop"}]}'
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
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto", stream=True),
            effective_config=provider.config,
        )
    ]

    completed = [chunk for chunk in chunks if chunk.completed_tool_calls]
    assert len(completed) == 1
    final_completed_chunk = completed[0]
    assert final_completed_chunk.finish_reason == "stop"
    assert final_completed_chunk.completed_tool_calls is not None
    assert len(final_completed_chunk.completed_tool_calls) == 1
    assert final_completed_chunk.completed_tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_sse_stream_flushes_completed_tool_calls_without_terminal_finish_reason(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}'
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
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto", stream=True),
            effective_config=provider.config,
        )
    ]

    completed = [chunk for chunk in chunks if chunk.completed_tool_calls]
    assert len(completed) == 1
    assert completed[0].raw_chunk is None
    assert completed[0].completed_tool_calls is not None
    assert len(completed[0].completed_tool_calls) == 1
    assert completed[0].completed_tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )

    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    ["length", "content_filter", "abort", "error"],
)
async def test_vllm_sse_stream_does_not_complete_tool_calls_on_non_tool_finish_reasons(
    sample_messages,
    sample_tools,
    finish_reason: str,
) -> None:
    sse_payload = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\""}}]},"finish_reason":"'
        + finish_reason
        + '"}]}'
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
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto", stream=True),
            effective_config=provider.config,
        )
    ]

    completed = [chunk for chunk in chunks if chunk.completed_tool_calls]
    assert not completed
    assert chunks[-1].finish_reason == finish_reason

    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_stream_aggregation_keeps_provider_raw_chunk_on_done_flush(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"id":"chunk_1","model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}'
        "\n\n"
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
    )
    client = Conduit(VLLMConfig(model="m"))
    await client._provider.http_client.aclose()
    client._provider.http_client = httpx.AsyncClient(transport=transport)

    response = await client.chat(
        messages=sample_messages,
        tools=sample_tools,
        tool_choice="auto",
        stream=True,
    )

    assert response.raw_response is not None
    assert response.raw_response.get("id") == "chunk_1"
    assert response.raw_response.get("event") is None

    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_sse_stream_emits_usage_only_terminal_chunk(sample_messages) -> None:
    sse_payload = (
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}'
        "\n\n"
        'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
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

    usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage is not None
    assert usage_chunks[0].usage.prompt_tokens == 1
    assert usage_chunks[0].usage.completion_tokens == 1
    assert usage_chunks[0].usage.total_tokens == 2

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
    assert len(completed) == 1
    assert completed[0].completed_tool_calls is not None
    assert len(completed[0].completed_tool_calls) == 1
    assert completed[0].completed_tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_assembles_tool_calls_with_stop_finish_reason(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{"}}]},"finish_reason":null}]}}'
        "\n\n"
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"type":"function","function":{"arguments":"\\"location\\":\\"NYC\\"}"}}]},"finish_reason":"stop"}]}}'
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
    assert len(completed) == 1
    final_completed_chunk = completed[0]
    assert final_completed_chunk.finish_reason == "stop"
    assert final_completed_chunk.completed_tool_calls is not None
    assert len(final_completed_chunk.completed_tool_calls) == 1
    assert final_completed_chunk.completed_tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_does_not_complete_tool_calls_when_native_reason_is_non_tool(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}}'
        "\n\n"
        'data: {"data":{"choices":[{"delta":{},"finish_reason":"stop","native_finish_reason":"length"}]}}'
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
    assert not completed
    assert chunks[-1].finish_reason == "stop"

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_does_not_flush_completed_tool_calls_when_only_native_non_tool_reason(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}}'
        "\n\n"
        'data: {"data":{"choices":[{"delta":{},"finish_reason":null,"native_finish_reason":"length"}]}}'
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
    assert not completed

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_sse_stream_flushes_completed_tool_calls_without_terminal_finish_reason(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}}'
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
    assert len(completed) == 1
    assert completed[0].raw_chunk is None
    assert completed[0].completed_tool_calls is not None
    assert len(completed[0].completed_tool_calls) == 1
    assert completed[0].completed_tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_stream_aggregation_keeps_provider_raw_chunk_on_done_flush(
    sample_messages,
    sample_tools,
) -> None:
    sse_payload = (
        'data: {"data":{"id":"chunk_1","model":"openai/gpt-4o-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\"}"}}]},"finish_reason":null}]}}'
        "\n\n"
        "data: [DONE]\n\n"
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=sse_payload, headers={"content-type": "text/event-stream"})
    )
    client = Conduit(OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"))
    await client._provider.http_client.aclose()
    client._provider.http_client = httpx.AsyncClient(transport=transport)

    response = await client.chat(
        messages=sample_messages,
        tools=sample_tools,
        tool_choice="auto",
        stream=True,
    )

    assert response.raw_response is not None
    assert response.raw_response.get("id") == "chunk_1"
    assert response.raw_response.get("event") is None

    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    ["length", "content_filter", "error"],
)
async def test_openrouter_sse_stream_does_not_complete_tool_calls_on_non_tool_finish_reasons(
    sample_messages,
    sample_tools,
    finish_reason: str,
) -> None:
    sse_payload = (
        'data: {"data":{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\":\\"NYC\\""}}]},"finish_reason":"'
        + finish_reason
        + '"}]}}'
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
    assert not completed
    assert chunks[-1].finish_reason == finish_reason

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
    assert len(usage_chunks) == 1
    assert usage_chunks[0].usage is not None
    assert usage_chunks[0].usage.prompt_tokens == 1
    assert usage_chunks[0].usage.completion_tokens == 1
    assert usage_chunks[0].usage.total_tokens == 2

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
    assert len(chunks[-1].completed_tool_calls) == 1
    assert chunks[-1].completed_tool_calls[0].name == "get_weather"
    assert chunks[-1].completed_tool_calls[0].arguments == {"location": "NYC"}
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 5
    assert chunks[-1].usage.completion_tokens == 2
    assert chunks[-1].usage.total_tokens == 7

    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_generate_ndjson_stream_normalizes_chunks(sample_messages) -> None:
    requested_path: str | None = None
    ndjson_payload = "\n".join(
        [
            '{"model":"qwen3","response":"Hel","done":false}',
            '{"model":"qwen3","response":"lo","done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2}',
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_path
        requested_path = request.url.path
        return httpx.Response(
            200,
            text=ndjson_payload,
            headers={"content-type": "application/x-ndjson"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(OllamaConfig(model="qwen3", raw=True), http_client=client)

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            ChatRequest(messages=sample_messages, stream=True),
            effective_config=provider.config,
        )
    ]

    assert requested_path == "/api/generate"
    assert "".join(chunk.content or "" for chunk in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.prompt_tokens == 4
    assert chunks[-1].usage.completion_tokens == 2
    assert chunks[-1].usage.total_tokens == 6

    await client.aclose()
