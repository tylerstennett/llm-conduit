from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from conduit.exceptions import StreamError
from conduit.models.messages import PartialToolCall
from conduit.tools.schema import ToolCall


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield SSE ``data:`` payload strings from an HTTP response stream.

    Comment lines (starting with ``:``) are skipped. Multi-line data
    fields are joined with newlines.

    Args:
        response: An open ``httpx.Response`` in streaming mode.

    Yields:
        str: The payload string for each SSE event.
    """
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


async def iter_ndjson(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed JSON objects from an NDJSON response stream.

    Args:
        response: An open ``httpx.Response`` in streaming mode.

    Yields:
        dict[str, Any]: Parsed JSON object for each non-empty line.

    Raises:
        StreamError: On malformed JSON or non-object payloads.
    """
    async for line in response.aiter_lines():
        if line is None:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise StreamError(f"invalid NDJSON line: {exc}") from exc
        if not isinstance(parsed, dict):
            raise StreamError("NDJSON event must decode to an object")
        yield parsed


def parse_openai_stream_tool_calls(
    raw_tool_calls: list[dict[str, Any]] | None,
) -> list[PartialToolCall] | None:
    """Parse streaming tool-call delta fragments from an OpenAI-format chunk.

    Args:
        raw_tool_calls: Raw ``tool_calls`` array from a stream delta.

    Returns:
        List of ``PartialToolCall`` objects, or ``None`` if empty.
    """
    if not raw_tool_calls:
        return None

    partials: list[PartialToolCall] = []
    for raw_call in raw_tool_calls:
        index = raw_call.get("index")
        if not isinstance(index, int):
            continue
        call_id = raw_call.get("id")
        if call_id is not None and not isinstance(call_id, str):
            call_id = None

        function = raw_call.get("function")
        name: str | None = None
        arguments_fragment: str | None = None
        if isinstance(function, dict):
            raw_name = function.get("name")
            if isinstance(raw_name, str):
                name = raw_name
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                arguments_fragment = raw_arguments

        partials.append(
            PartialToolCall(
                index=index,
                id=call_id,
                name=name,
                arguments_fragment=arguments_fragment,
            )
        )

    return partials or None


class ToolCallChunkAccumulator:
    """Assembles streamed OpenAI-style tool-call fragments into complete calls."""

    def __init__(self) -> None:
        self._items: dict[int, dict[str, Any]] = {}

    def ingest(self, partial: PartialToolCall) -> None:
        """Accumulate a single tool-call delta fragment.

        Args:
            partial: A streaming tool-call fragment.
        """
        item = self._items.setdefault(
            partial.index,
            {"id": None, "name": None, "arguments_parts": []},
        )
        if partial.id:
            item["id"] = partial.id
        if partial.name:
            item["name"] = partial.name
        if partial.arguments_fragment:
            item["arguments_parts"].append(partial.arguments_fragment)

    def completed_calls(self) -> list[ToolCall]:
        """Return all fully-assembled tool calls accumulated so far.

        Returns:
            List of ``ToolCall`` objects (may be empty).
        """
        calls: list[ToolCall] = []
        for index in sorted(self._items):
            entry = self._items[index]
            call_id = entry["id"] or f"call_{index}"
            name = entry["name"] or ""
            if not name:
                continue
            arguments_text = "".join(entry["arguments_parts"])
            if not arguments_text.strip():
                arguments: dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(arguments_text)
                except json.JSONDecodeError:
                    parsed = {}
                arguments = parsed if isinstance(parsed, dict) else {}
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return calls

