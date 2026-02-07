from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from conduit.exceptions import StreamError
from conduit.models.messages import PartialToolCall


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield SSE data payload strings from an HTTP response stream."""
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
    """Yield parsed JSON objects from an NDJSON response stream."""
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

