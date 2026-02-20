from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from conduit.client import Conduit
from conduit.config import VLLMConfig
from conduit.models.messages import (
    ChatRequest,
    ChatResponseChunk,
    Message,
    PartialToolCall,
    Role,
    TextPart,
    UsageStats,
)
from conduit.tools.schema import ToolCall


@pytest.mark.asyncio
async def test_chat_events_emits_deterministic_order() -> None:
    client = Conduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        request: ChatRequest,
        *,
        effective_config: object,
    ) -> AsyncIterator[ChatResponseChunk]:
        del request, effective_config
        yield ChatResponseChunk(
            content="Hel",
            tool_calls=[
                PartialToolCall(
                    index=0,
                    id="call_1",
                    name="get_weather",
                    arguments_fragment="{",
                )
            ],
            raw_chunk={"model": "m"},
        )
        yield ChatResponseChunk(
            content="lo",
            tool_calls=[
                PartialToolCall(
                    index=0,
                    arguments_fragment='"location":"NYC"}',
                )
            ],
            completed_tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments={"location": "NYC"},
                )
            ],
            usage=UsageStats(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="tool_calls",
            raw_chunk={"model": "m"},
        )

    client._provider.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    events = [
        event
        async for event in client.chat_events(
            messages=[Message(role=Role.USER, content=[TextPart(text="hi")])]
        )
    ]

    assert [event.type for event in events] == [
        "text_delta",
        "tool_call_delta",
        "text_delta",
        "tool_call_delta",
        "tool_call_completed",
        "usage",
        "finish",
    ]
    assert events[0].text == "Hel"
    assert events[2].text == "lo"
    assert events[4].tool_call == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )
    assert events[5].usage == UsageStats(
        prompt_tokens=1, completion_tokens=1, total_tokens=2
    )
    assert events[6].finish_reason == "tool_calls"
    await client.aclose()


@pytest.mark.asyncio
async def test_chat_events_emits_error_then_raises() -> None:
    client = Conduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        request: ChatRequest,
        *,
        effective_config: object,
    ) -> AsyncIterator[ChatResponseChunk]:
        del request, effective_config
        yield ChatResponseChunk(content="partial")
        raise RuntimeError("boom")

    client._provider.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    events = []
    with pytest.raises(RuntimeError, match="boom"):
        async for event in client.chat_events(
            messages=[Message(role=Role.USER, content=[TextPart(text="hi")])]
        ):
            events.append(event)

    assert [event.type for event in events] == ["text_delta", "error"]
    assert events[-1].error == "boom"
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_event_aggregation_reconstructs_chat_response() -> None:
    client = Conduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        request: ChatRequest,
        *,
        effective_config: object,
    ) -> AsyncIterator[ChatResponseChunk]:
        del request, effective_config
        yield ChatResponseChunk(content="Hel", raw_chunk={"model": "model-a"})
        yield ChatResponseChunk(
            content="lo",
            completed_tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"})
            ],
            finish_reason="stop",
            usage=UsageStats(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            raw_chunk={"model": "model-a", "id": "chunk-final"},
        )

    client._provider.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    response = await client.chat(
        messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
        stream=True,
    )

    assert response.content == "Hello"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0] == ToolCall(
        id="call_1", name="get_weather", arguments={"location": "NYC"}
    )
    assert response.finish_reason == "stop"
    assert response.usage == UsageStats(
        prompt_tokens=1, completion_tokens=1, total_tokens=2
    )
    assert response.raw_response == {"model": "model-a", "id": "chunk-final"}
    assert response.model == "model-a"
    assert response.provider == "vllm"
    await client.aclose()
