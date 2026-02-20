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

    @classmethod
    def from_pydantic(
        cls,
        name: str,
        description: str,
        model: type[BaseModel],
        *,
        strict: bool | None = None,
    ) -> ToolDefinition:
        """Create a ``ToolDefinition`` from a Pydantic model class.

        The model's JSON schema (via ``model_json_schema()``) is used as
        the ``parameters`` value.  Pydantic v2 always produces a schema
        with ``"type": "object"`` at the root, so the existing validator
        is satisfied automatically.

        Args:
            name: Tool name.
            description: Human-readable tool description.
            model: Pydantic ``BaseModel`` subclass whose schema defines
                the tool parameters.
            strict: Optional strict-mode flag for providers that support it.

        Returns:
            A new ``ToolDefinition`` instance.
        """
        return cls(
            name=name,
            description=description,
            parameters=model.model_json_schema(),
            strict=strict,
        )

    def to_openai_tool(self, *, include_strict: bool = False) -> dict[str, Any]:
        """Serialize to the OpenAI function-calling tool format.

        Args:
            include_strict: When ``True`` and ``self.strict`` is set,
                include the ``strict`` key in the function payload.

        Returns:
            A dict matching the OpenAI ``{"type": "function", "function": ...}``
            schema.
        """
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
    """Normalize provider tool-argument payloads into dicts.

    Accepts JSON strings, dicts, or ``None`` and returns a dict in all
    cases.

    Args:
        raw_arguments: Raw arguments from a provider response (string,
            dict, or ``None``).

    Returns:
        Parsed arguments dict (empty dict for ``None`` or blank strings).

    Raises:
        ToolSchemaError: If the string is not valid JSON or does not
            decode to an object.
    """
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
