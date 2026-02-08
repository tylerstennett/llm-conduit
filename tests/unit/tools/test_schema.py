from __future__ import annotations

import pytest
from pydantic import ValidationError

from conduit.tools.schema import ToolDefinition


def test_tool_definition_defaults_to_valid_empty_object_schema() -> None:
    tool = ToolDefinition(name="foo", description="bar")

    assert tool.parameters == {
        "type": "object",
        "properties": {},
    }
    assert tool.strict is None


def test_tool_definition_rejects_explicit_empty_parameters_object() -> None:
    with pytest.raises(ValidationError, match="type='object'"):
        ToolDefinition(name="foo", description="bar", parameters={})


def test_to_openai_tool_omits_strict_by_default() -> None:
    tool = ToolDefinition(name="foo", description="bar", strict=True)

    payload = tool.to_openai_tool()

    assert "strict" not in payload["function"]


def test_to_openai_tool_includes_strict_true_when_enabled() -> None:
    tool = ToolDefinition(name="foo", description="bar", strict=True)

    payload = tool.to_openai_tool(include_strict=True)

    assert payload["function"]["strict"] is True


def test_to_openai_tool_includes_strict_false_when_enabled() -> None:
    tool = ToolDefinition(name="foo", description="bar", strict=False)

    payload = tool.to_openai_tool(include_strict=True)

    assert payload["function"]["strict"] is False
