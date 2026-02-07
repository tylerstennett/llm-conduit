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


def test_tool_definition_rejects_explicit_empty_parameters_object() -> None:
    with pytest.raises(ValidationError, match="type='object'"):
        ToolDefinition(name="foo", description="bar", parameters={})
