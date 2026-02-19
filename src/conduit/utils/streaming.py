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
    """Decide whether accumulated tool-call fragments should be completed.

    The decision follows a three-tier priority:

    1. **native_finish_reason present** (OpenRouter dual-reason): complete only
       if the native reason is an explicit tool reason (``tool_calls`` or
       ``function_call``) *and* we actually saw tool-call deltas during the
       stream.
    2. **finish_reason is ``tool_calls`` / ``function_call``**: always complete
       — the provider explicitly signalled a tool invocation.
    3. **finish_reason is ``stop``**: complete only if we saw tool-call deltas.
       This handles providers that report ``stop`` even when the response
       contains tool calls.
    4. **Any other reason** (e.g. ``length``): never complete.
    """
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
