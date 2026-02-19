from __future__ import annotations

import json
from typing import Any

from conduit.exceptions import ConfigValidationError
from conduit.models.messages import ImageUrlPart, Message, Role, TextPart, UsageStats
from conduit.tools.schema import ToolCall, ToolDefinition, parse_tool_arguments


def normalize_stop(stop: list[str] | str | None) -> list[str] | str | None:
    if stop is None:
        return None
    if isinstance(stop, list):
        return stop if stop else None
    return stop


def tool_choice_to_openai_payload(
    tool_choice: str | dict[str, Any] | None,
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise ConfigValidationError(
                "tool_choice string must be one of: auto, none, required"
            )
        return tool_choice
    if isinstance(tool_choice, dict):
        return tool_choice
    raise ConfigValidationError("tool_choice must be a string, dict, or None")


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        payload: dict[str, Any] = {"role": message.role.value}
        if message.name:
            payload["name"] = message.name

        content = to_openai_message_content(message)
        if content is not None:
            payload["content"] = content

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]

        if message.role is Role.TOOL and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id

        output.append(payload)
    return output


def to_openai_message_content(message: Message) -> list[dict[str, Any]] | None:
    content_items = _serialize_openai_content_parts(message.content)
    if content_items:
        return content_items
    return None


def _serialize_openai_content_parts(
    parts: list[TextPart | ImageUrlPart | dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not parts:
        return []

    output: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            output.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, ImageUrlPart):
            output.append(
                {
                    "type": "image_url",
                    "image_url": {"url": part.url},
                }
            )
            continue

        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                output.append(dict(part))
                continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                nested_url = image_url.get("url")
                if isinstance(nested_url, str):
                    output.append(dict(part))
                    continue

        output.append(part)
    return output


def tool_definitions_to_openai(
    tools: list[ToolDefinition] | None,
    *,
    include_strict: bool = False,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [tool.to_openai_tool(include_strict=include_strict) for tool in tools]


def ensure_tool_strict_supported(
    tools: list[ToolDefinition] | None,
    *,
    provider_name: str,
    supports_tool_strict: bool,
) -> None:
    if supports_tool_strict or not tools:
        return

    strict_tool_names = sorted(tool.name for tool in tools if tool.strict is True)
    if strict_tool_names:
        joined = ", ".join(strict_tool_names)
        raise ConfigValidationError(
            f"{provider_name} does not support strict tool schemas; "
            f"strict=true was set for tool(s): {joined}"
        )


def parse_openai_tool_calls(
    raw_calls: list[dict[str, Any]] | None,
) -> list[ToolCall] | None:
    if not raw_calls:
        return None

    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        function = raw_call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        arguments = parse_tool_arguments(function.get("arguments"))
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))

    return parsed or None


def parse_usage(usage: dict[str, Any] | None) -> UsageStats | None:
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return UsageStats(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=(
            completion_tokens if isinstance(completion_tokens, int) else None
        ),
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
    )


def extract_openai_message_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None
