from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from conduit.tools.schema import ToolCall, ToolDefinition


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RequestContext(BaseModel):
    """Run-scoped metadata for one invocation."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextPart(BaseModel):
    """Text content part."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ImageUrlPart(BaseModel):
    """Image URL content part."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    url: str


ContentPart = TextPart | ImageUrlPart | dict[str, Any]


class Message(BaseModel):
    """Canonical chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


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
    context: RequestContext | None = None
    runtime_overrides: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    """Typed streaming event."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "text_delta",
        "tool_call_delta",
        "tool_call_completed",
        "usage",
        "finish",
        "error",
    ]
    text: str | None = None
    tool_call: PartialToolCall | ToolCall | None = None
    usage: UsageStats | None = None
    finish_reason: str | None = None
    error: str | None = None
    raw: dict[str, Any] | None = None


class StreamEventAccumulator:
    """Reconstructs a ChatResponse from stream events."""

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._finish_reason: str | None = None
        self._usage: UsageStats | None = None
        self._raw_response: dict[str, Any] | None = None
        self._model: str | None = None

    def ingest(self, event: StreamEvent) -> None:
        if event.type == "text_delta" and event.text:
            self._content_parts.append(event.text)

        if event.type == "tool_call_completed" and isinstance(event.tool_call, ToolCall):
            self._tool_calls.append(event.tool_call)

        if event.type == "finish" and event.finish_reason is not None:
            self._finish_reason = event.finish_reason

        if event.type == "usage" and event.usage is not None:
            self._usage = event.usage

        if event.raw is not None:
            self._raw_response = event.raw
            raw_model = event.raw.get("model")
            if isinstance(raw_model, str):
                self._model = raw_model

    def to_response(self, *, provider: str | None = None) -> ChatResponse:
        content = "".join(self._content_parts) if self._content_parts else None
        tool_calls = self._tool_calls if self._tool_calls else None
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._finish_reason,
            usage=self._usage,
            raw_response=self._raw_response,
            model=self._model,
            provider=provider,
        )


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
