from __future__ import annotations

import pytest

from conduit.models.messages import Message, Role
from conduit.tools.schema import ToolDefinition


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are helpful."),
        Message(role=Role.USER, content="What is the weather in New York?"),
    ]


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        )
    ]

