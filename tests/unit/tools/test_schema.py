from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

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


# -- from_pydantic --------------------------------------------------------


class _WeatherArgs(BaseModel):
    location: str
    unit: Literal["celsius", "fahrenheit"] = "celsius"


def test_from_pydantic_produces_valid_tool_definition() -> None:
    tool = ToolDefinition.from_pydantic(
        name="get_weather",
        description="Get weather for a location",
        model=_WeatherArgs,
    )

    assert tool.name == "get_weather"
    assert tool.description == "Get weather for a location"
    assert tool.parameters["type"] == "object"
    assert "location" in tool.parameters["properties"]
    assert "unit" in tool.parameters["properties"]
    assert tool.strict is None


def test_from_pydantic_required_fields() -> None:
    tool = ToolDefinition.from_pydantic(
        name="get_weather",
        description="Get weather",
        model=_WeatherArgs,
    )

    assert "location" in tool.parameters["required"]


def test_from_pydantic_passes_strict_through() -> None:
    tool = ToolDefinition.from_pydantic(
        name="get_weather",
        description="Get weather",
        model=_WeatherArgs,
        strict=True,
    )

    assert tool.strict is True


def test_from_pydantic_round_trips_through_openai_format() -> None:
    tool = ToolDefinition.from_pydantic(
        name="get_weather",
        description="Get weather",
        model=_WeatherArgs,
    )

    payload = tool.to_openai_tool()

    assert payload["type"] == "function"
    assert payload["function"]["name"] == "get_weather"
    assert payload["function"]["parameters"]["type"] == "object"


class _Empty(BaseModel):
    pass


def test_from_pydantic_empty_model() -> None:
    tool = ToolDefinition.from_pydantic(
        name="noop",
        description="Does nothing",
        model=_Empty,
    )

    assert tool.parameters["type"] == "object"
    assert tool.parameters["properties"] == {}


class _Nested(BaseModel):
    class Address(BaseModel):
        street: str
        city: str

    name: str
    address: Address


def test_from_pydantic_nested_model() -> None:
    tool = ToolDefinition.from_pydantic(
        name="create_contact",
        description="Create a contact",
        model=_Nested,
    )

    assert "address" in tool.parameters["properties"]
    assert tool.parameters["type"] == "object"


class _Described(BaseModel):
    query: str = Field(description="The search query")


def test_from_pydantic_preserves_field_descriptions() -> None:
    tool = ToolDefinition.from_pydantic(
        name="search",
        description="Search something",
        model=_Described,
    )

    assert tool.parameters["properties"]["query"]["description"] == "The search query"
