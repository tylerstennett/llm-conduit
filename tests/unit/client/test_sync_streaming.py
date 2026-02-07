from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import pytest

from conduit.client import SyncConduit
from conduit.config import VLLMConfig
from conduit.models.messages import ChatResponseChunk, Message
from conduit.tools.schema import ToolDefinition


def test_sync_chat_stream_yields_chunks_incrementally() -> None:
    client = SyncConduit(VLLMConfig(model="m"))

    async def fake_chat_stream(
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, object] | None = None,
        config_overrides: dict[str, object] | None = None,
    ) -> AsyncIterator[ChatResponseChunk]:
        del messages, tools, tool_choice, config_overrides
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
