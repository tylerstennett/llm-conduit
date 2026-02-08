from __future__ import annotations

from conduit.models.messages import ChatResponseChunk


def should_emit_stream_chunk(chunk: ChatResponseChunk) -> bool:
    return (
        chunk.content is not None
        or chunk.tool_calls is not None
        or chunk.completed_tool_calls is not None
        or chunk.finish_reason is not None
        or chunk.usage is not None
    )
