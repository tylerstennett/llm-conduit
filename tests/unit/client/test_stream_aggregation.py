from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from conduit.client import Conduit
from conduit.config import VLLMConfig
from conduit.models.messages import (
    ChatRequest,
    ChatResponseChunk,
    Message,
    Role,
    TextPart,
    UsageStats,
)
from conduit.tools.schema import ToolCall


@pytest.mark.asyncio
async def test_chat_stream_true_aggregates_chunks_into_chat_response() -> None:
    client = Conduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        request: ChatRequest,
        *,
        effective_config: object,
    ) -> AsyncIterator[ChatResponseChunk]:
        del request, effective_config
        yield ChatResponseChunk(content="Hel")
        yield ChatResponseChunk(
            content="lo",
            usage=UsageStats(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            raw_chunk={"model": "midpoint"},
        )
        yield ChatResponseChunk(
            completed_tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"})
            ],
            finish_reason="tool_calls",
            raw_chunk={"model": "final-model", "id": "chunk-3"},
        )

    client._provider.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    response = await client.chat(
        messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
        stream=True,
    )
    await client.aclose()

    assert response.content == "Hello"
    assert response.finish_reason == "tool_calls"
    assert response.usage == UsageStats(
        prompt_tokens=1, completion_tokens=2, total_tokens=3
    )
    assert response.model == "final-model"
    assert response.raw_response == {"model": "final-model", "id": "chunk-3"}
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )
    assert response.provider == "vllm"


@pytest.mark.asyncio
async def test_chat_stream_true_with_no_chunks_returns_empty_response() -> None:
    client = Conduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        request: ChatRequest,
        *,
        effective_config: object,
    ) -> AsyncIterator[ChatResponseChunk]:
        del request, effective_config
        if False:
            yield ChatResponseChunk(content="never")

    client._provider.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    response = await client.chat(
        messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
        stream=True,
    )
    await client.aclose()

    assert response.content is None
    assert response.finish_reason is None
    assert response.usage is None
    assert response.tool_calls is None
    assert response.raw_response is None
    assert response.model is None
    assert response.provider == "vllm"
