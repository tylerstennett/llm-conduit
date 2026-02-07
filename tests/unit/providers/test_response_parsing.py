from __future__ import annotations

import httpx
import pytest

from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.models.messages import ChatRequest, Message, Role
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openrouter import OpenRouterProvider
from conduit.providers.vllm import VLLMProvider
from conduit.tools.schema import ToolCall


@pytest.mark.asyncio
async def test_vllm_parses_chat_response(sample_messages, sample_tools) -> None:
    payload = {
        "id": "chatcmpl-1",
        "model": "model-a",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"New York"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = httpx.AsyncClient(transport=transport)
    provider = VLLMProvider(VLLMConfig(model="m"), http_client=client)

    response = await provider.chat(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=provider.config,
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].arguments["location"] == "New York"
    assert response.usage is not None
    assert response.usage.total_tokens == 14

    await client.aclose()


@pytest.mark.asyncio
async def test_openrouter_parses_chat_response(sample_messages, sample_tools) -> None:
    payload = {
        "id": "or-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        },
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = httpx.AsyncClient(transport=transport)
    provider = OpenRouterProvider(
        OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"),
        http_client=client,
    )

    response = await provider.chat(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=provider.config,
    )

    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 4

    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_parses_chat_response(sample_messages, sample_tools) -> None:
    payload = {
        "model": "qwen3",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "New York"},
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 7,
        "eval_count": 2,
    }

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(OllamaConfig(model="qwen3"), http_client=client)

    response = await provider.chat(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=provider.config,
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "ollama_call_0"
    assert response.tool_calls[0].arguments["location"] == "New York"
    assert response.usage is not None
    assert response.usage.total_tokens == 9

    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_generate_mode_parses_response(sample_messages) -> None:
    requested_path: str | None = None
    payload = {
        "model": "qwen3",
        "response": "generated text",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 4,
        "eval_count": 3,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_path
        requested_path = request.url.path
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(OllamaConfig(model="qwen3", raw=True), http_client=client)

    response = await provider.chat(
        ChatRequest(messages=sample_messages),
        effective_config=provider.config,
    )

    assert requested_path == "/api/generate"
    assert response.content == "generated text"
    assert response.finish_reason == "stop"
    assert response.tool_calls is None
    assert response.usage is not None
    assert response.usage.total_tokens == 7

    await client.aclose()


def test_ollama_round_trip_tool_name_mapping(sample_tools) -> None:
    assistant_message = Message(
        role=Role.ASSISTANT,
        tool_calls=[ToolCall(id="call_abc", name="get_weather", arguments={"location": "NYC"})],
    )
    tool_result_message = Message(
        role=Role.TOOL,
        tool_call_id="call_abc",
        content="{\"temp\": 72}",
    )

    provider = OllamaProvider(OllamaConfig(model="qwen3"))
    body = provider.build_request_body(
        ChatRequest(messages=[assistant_message, tool_result_message], tools=sample_tools),
        effective_config=provider.config,
        stream=False,
    )

    assert body["messages"][1]["tool_name"] == "get_weather"
