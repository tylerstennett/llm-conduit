from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from conduit.tools.schema import ToolCall, ToolDefinition


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Canonical chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    images: list[str] | None = None


class UsageStats(BaseModel):
    """Canonical token usage information."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class PartialToolCall(BaseModel):
    """Incremental tool call chunk while streaming."""

    model_config = ConfigDict(extra="forbid")

    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str | None = None


class ChatResponseChunk(BaseModel):
    """Canonical stream chunk."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[PartialToolCall] | None = None
    completed_tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: UsageStats | None = None
    raw_chunk: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Canonical non-stream chat response."""

    model_config = ConfigDict(extra="forbid")

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: UsageStats | None = None
    raw_response: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None


class ChatRequest(BaseModel):
    """Canonical request passed into providers."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    config_overrides: dict[str, Any] | None = None


class ToolCallChunkAccumulator:
    """Assembles streamed OpenAI-style tool call fragments."""

    def __init__(self) -> None:
        self._items: dict[int, dict[str, Any]] = {}

    def ingest(self, partial: PartialToolCall) -> None:
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
