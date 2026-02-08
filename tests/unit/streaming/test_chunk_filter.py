from __future__ import annotations

from conduit.models.messages import ChatResponseChunk, PartialToolCall, UsageStats
from conduit.tools.schema import ToolCall
from conduit.utils import should_emit_stream_chunk


def test_should_emit_stream_chunk_false_for_empty_chunk() -> None:
    chunk = ChatResponseChunk()
    assert should_emit_stream_chunk(chunk) is False


def test_should_emit_stream_chunk_true_for_usage_only_chunk() -> None:
    chunk = ChatResponseChunk(usage=UsageStats(total_tokens=4))
    assert should_emit_stream_chunk(chunk) is True


def test_should_emit_stream_chunk_true_for_tool_data() -> None:
    partial_chunk = ChatResponseChunk(tool_calls=[PartialToolCall(index=0, name="tool")])
    completed_chunk = ChatResponseChunk(
        completed_tool_calls=[ToolCall(id="call_1", name="tool", arguments={})]
    )
    assert should_emit_stream_chunk(partial_chunk) is True
    assert should_emit_stream_chunk(completed_chunk) is True
