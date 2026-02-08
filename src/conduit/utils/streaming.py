from __future__ import annotations

from conduit.models.messages import ChatResponseChunk

_EXPLICIT_TOOL_CALL_FINISH_REASONS = frozenset({"tool_calls", "function_call"})


def _normalize_finish_reason(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    normalized = finish_reason.strip().lower()
    return normalized or None


def should_complete_tool_calls(
    *,
    finish_reason: str | None,
    saw_tool_call_delta: bool,
    native_finish_reason: str | None = None,
) -> bool:
    normalized_native_reason = _normalize_finish_reason(native_finish_reason)
    if normalized_native_reason is not None:
        if normalized_native_reason in _EXPLICIT_TOOL_CALL_FINISH_REASONS:
            return saw_tool_call_delta
        return False

    normalized_reason = _normalize_finish_reason(finish_reason)
    if normalized_reason in _EXPLICIT_TOOL_CALL_FINISH_REASONS:
        return True

    if normalized_reason == "stop":
        return saw_tool_call_delta

    return False


def should_emit_stream_chunk(chunk: ChatResponseChunk) -> bool:
    return (
        chunk.content is not None
        or chunk.tool_calls is not None
        or chunk.completed_tool_calls is not None
        or chunk.finish_reason is not None
        or chunk.usage is not None
    )
