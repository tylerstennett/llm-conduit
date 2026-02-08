from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import pytest

from conduit.client import SyncConduit
from conduit.config import VLLMConfig
from conduit.models.messages import (
    ChatResponse,
    ChatResponseChunk,
    Message,
    RequestContext,
    StreamEvent,
)
from conduit.tools.schema import ToolDefinition


def test_sync_chat_stream_yields_chunks_incrementally() -> None:
    client = SyncConduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        config_overrides: dict[str, object] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, object] | None = None,
    ) -> AsyncIterator[ChatResponseChunk]:
        del messages, tools, tool_choice, config_overrides, context, runtime_overrides
        yield ChatResponseChunk(content="first")
        await asyncio.sleep(0.2)
        yield ChatResponseChunk(content="second")

    client._async_client.chat_stream = fake_chat_stream  # type: ignore[method-assign]

    start = time.perf_counter()
    stream = client.chat_stream(messages=[])
    first = next(stream)
    elapsed = time.perf_counter() - start

    assert first.content == "first"
    assert elapsed < 0.15

    second = next(stream)
    assert second.content == "second"
    with pytest.raises(StopIteration):
        next(stream)

    client.close()


def test_sync_chat_reuses_single_event_loop_across_calls() -> None:
    client = SyncConduit(VLLMConfig(model="m"))
    loop_ids: list[int] = []

    async def fake_chat(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        stream: bool = False,
        config_overrides: dict[str, object] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, object] | None = None,
    ) -> ChatResponse:
        del (
            messages,
            tools,
            tool_choice,
            stream,
            config_overrides,
            context,
            runtime_overrides,
        )
        loop_ids.append(id(asyncio.get_running_loop()))
        return ChatResponse(content="ok")

    client._async_client.chat = fake_chat  # type: ignore[method-assign]

    first = client.chat(messages=[])
    second = client.chat(messages=[])

    assert first.content == "ok"
    assert second.content == "ok"
    assert len(set(loop_ids)) == 1

    client.close()


def test_sync_context_manager_closes_client() -> None:
    with SyncConduit(VLLMConfig(model="m")) as client:
        pass

    with pytest.raises(RuntimeError, match="SyncConduit client is closed"):
        client.chat(messages=[])


def test_sync_close_is_idempotent() -> None:
    client = SyncConduit(VLLMConfig(model="m"))
    client.close()
    client.close()


def test_sync_chat_events_yields_events_incrementally() -> None:
    client = SyncConduit(VLLMConfig(model="m"))

    async def fake_chat_events(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        config_overrides: dict[str, object] | None = None,
        context: RequestContext | None = None,
        runtime_overrides: dict[str, object] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del messages, tools, tool_choice, config_overrides, context, runtime_overrides
        yield StreamEvent(type="text_delta", text="first")
        await asyncio.sleep(0.2)
        yield StreamEvent(type="finish", finish_reason="stop")

    client._async_client.chat_events = fake_chat_events  # type: ignore[method-assign]

    start = time.perf_counter()
    stream = client.chat_events(messages=[])
    first = next(stream)
    elapsed = time.perf_counter() - start

    assert first.type == "text_delta"
    assert elapsed < 0.15

    second = next(stream)
    assert second.type == "finish"
    with pytest.raises(StopIteration):
        next(stream)

    client.close()
