from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from conduit.exceptions import ToolSchemaError


def _default_tool_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
    }


class ToolDefinition(BaseModel):
    """Canonical function tool definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=_default_tool_parameters,
        validate_default=True,
    )
    strict: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("tool name must not be empty")
        return stripped

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("tool parameters root must set type='object'")
        return value

    def to_openai_tool(self, *, include_strict: bool = False) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if include_strict and self.strict is not None:
            function["strict"] = self.strict

        return {
            "type": "function",
            "function": function,
        }


class ToolCall(BaseModel):
    """Canonical parsed tool call."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def parse_tool_arguments(raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize provider tool argument payloads into dictionaries."""
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ToolSchemaError(f"failed to parse tool arguments: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ToolSchemaError("tool arguments must decode to a JSON object")
        return parsed
    raise ToolSchemaError("unsupported tool arguments type")
