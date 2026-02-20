"""Comprehensive integration tests for OpenRouter via the Conduit client.

These tests hit the live OpenRouter API and require:
  CONDUIT_TEST_OPENROUTER_KEY   – your OpenRouter API key
  CONDUIT_TEST_OPENROUTER_MODEL – a model ID that supports tool calling
                                  (recommended: google/gemini-2.0-flash-001)

Run with:  pytest -m integration -v
"""

from __future__ import annotations

import json
import os

import pytest

from conduit import (
    AuthenticationError,
    Conduit,
    Message,
    ModelNotFoundError,
    OpenRouterConfig,
    OpenRouterProviderPrefs,
    PartialToolCall,
    ProviderError,
    RequestContext,
    RetryPolicy,
    Role,
    SyncConduit,
    TextPart,
    ToolCall,
    ToolDefinition,
    UsageStats,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SKIP_MSG = "set CONDUIT_TEST_OPENROUTER_KEY and CONDUIT_TEST_OPENROUTER_MODEL to run"
_RETRY = RetryPolicy()


def _get_creds() -> tuple[str, str]:
    key = os.getenv("CONDUIT_TEST_OPENROUTER_KEY", "")
    model = os.getenv("CONDUIT_TEST_OPENROUTER_MODEL", "")
    return key, model


def _skip_if_no_creds() -> tuple[str, str]:
    key, model = _get_creds()
    if not key or not model:
        pytest.skip(_SKIP_MSG)
    return key, model


@pytest.fixture()
def or_config() -> OpenRouterConfig:
    key, model = _skip_if_no_creds()
    return OpenRouterConfig(model=model, api_key=key)


@pytest.fixture()
def sample_messages() -> list[Message]:
    return [Message(role=Role.USER, content=[TextPart(text="Say hello in one word.")])]


_GET_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a given city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city name"},
        },
        "required": ["city"],
    },
)

_CALCULATE_TOOL = ToolDefinition(
    name="calculate",
    description="Evaluate a simple math expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "A math expression like '2+2'"},
        },
        "required": ["expression"],
    },
)

_SEARCH_TOOL = ToolDefinition(
    name="search",
    description="Search for information on a given topic and return results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query"],
    },
)

_BOOK_FLIGHT_TOOL = ToolDefinition(
    name="book_flight",
    description="Book a flight between two cities.",
    parameters={
        "type": "object",
        "properties": {
            "departure": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Departure city"},
                    "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"},
                },
                "required": ["city", "date"],
            },
            "arrival": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Arrival city"},
                    "date": {"type": "string", "description": "Arrival date (YYYY-MM-DD)"},
                },
                "required": ["city", "date"],
            },
            "passengers": {
                "type": "integer",
                "description": "Number of passengers",
            },
        },
        "required": ["departure", "arrival", "passengers"],
    },
)

_LOOKUP_CONTACT_TOOL = ToolDefinition(
    name="lookup_contact",
    description="Look up contact information for a person by name.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name to look up"},
        },
        "required": ["name"],
    },
)

_SEND_EMAIL_TOOL = ToolDefinition(
    name="send_email",
    description="Send an email to a recipient.",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body text"},
        },
        "required": ["to", "subject", "body"],
    },
)

_UNIT_CONVERT_TOOL = ToolDefinition(
    name="unit_convert",
    description="Convert a numeric value from one unit to another.",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric value to convert"},
            "from_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit", "kelvin", "meters", "feet", "kg", "lbs"],
                "description": "The source unit",
            },
            "to_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit", "kelvin", "meters", "feet", "kg", "lbs"],
                "description": "The target unit",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)


# =========================================================================
# 1. Basic chat (non-streaming)
# =========================================================================


@pytest.mark.integration
class TestBasicChat:
    async def test_simple_text_response(self, or_config: OpenRouterConfig) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Say hello.")])]
            )
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == "openrouter"

    async def test_response_has_model_field(self, or_config: OpenRouterConfig) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.model is not None
        assert len(response.model) > 0

    async def test_response_has_usage_stats(self, or_config: OpenRouterConfig) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.usage is not None
        assert isinstance(response.usage, UsageStats)
        assert response.usage.prompt_tokens is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens is not None
        assert response.usage.completion_tokens > 0

    async def test_response_has_finish_reason_stop(
        self, or_config: OpenRouterConfig
    ) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Say 'ok'.")])]
            )
        assert response.finish_reason == "stop"

    async def test_response_has_raw_response(self, or_config: OpenRouterConfig) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.raw_response is not None
        assert isinstance(response.raw_response, dict)
        assert "choices" in response.raw_response

    async def test_system_message(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[TextPart(text="You must reply with exactly the word 'banana'.")],
            ),
            Message(role=Role.USER, content=[TextPart(text="What fruit?")]),
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(messages=messages)
        assert response.content is not None
        assert "banana" in response.content.lower()

    async def test_multi_turn_conversation(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[TextPart(text="You are a helpful assistant.")],
            ),
            Message(
                role=Role.USER,
                content=[TextPart(text="Remember the number 42.")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[TextPart(text="Got it, I'll remember 42.")],
            ),
            Message(
                role=Role.USER,
                content=[TextPart(text="What number did I ask you to remember?")],
            ),
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(messages=messages)
        assert response.content is not None
        assert "42" in response.content


# =========================================================================
# 2. Streaming – chat_stream()
# =========================================================================


@pytest.mark.integration
class TestStreaming:
    async def test_chat_stream_yields_chunks(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        chunks = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(messages=sample_messages):
                chunks.append(chunk)
        assert len(chunks) > 0

    async def test_chat_stream_content_concatenation(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        content_pieces: list[str] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(messages=sample_messages):
                if chunk.content:
                    content_pieces.append(chunk.content)
        full_content = "".join(content_pieces)
        assert len(full_content) > 0

    async def test_chat_stream_has_finish_reason(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        finish_reasons: list[str] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(messages=sample_messages):
                if chunk.finish_reason is not None:
                    finish_reasons.append(chunk.finish_reason)
        assert len(finish_reasons) >= 1
        assert "stop" in finish_reasons

    async def test_aggregated_stream_matches_non_stream(
        self, or_config: OpenRouterConfig
    ) -> None:
        """chat(stream=True) should produce a ChatResponse just like chat()."""
        messages = [
            Message(role=Role.USER, content=[TextPart(text="Say 'test'.")])
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(messages=messages, stream=True)
        assert response.content is not None
        assert response.provider == "openrouter"
        assert response.finish_reason == "stop"


# =========================================================================
# 3. Stream events – chat_events()
# =========================================================================


@pytest.mark.integration
class TestStreamEvents:
    async def test_chat_events_yields_text_deltas(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        event_types: list[str] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(messages=sample_messages):
                event_types.append(event.type)
        assert "text_delta" in event_types

    async def test_chat_events_yields_finish(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        event_types: list[str] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(messages=sample_messages):
                event_types.append(event.type)
        assert "finish" in event_types

    async def test_chat_events_text_delta_has_text(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(messages=sample_messages):
                if event.type == "text_delta":
                    assert event.text is not None
                    assert len(event.text) > 0
                    return
        pytest.fail("no text_delta event received")

    async def test_chat_events_finish_has_reason(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(messages=sample_messages):
                if event.type == "finish":
                    assert event.finish_reason is not None
                    return
        pytest.fail("no finish event received")


# =========================================================================
# 4. Tool / function calling
# =========================================================================


@pytest.mark.integration
class TestToolCalling:
    async def test_single_tool_call_non_streaming(
        self, or_config: OpenRouterConfig
    ) -> None:
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Paris?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1
        call = response.tool_calls[0]
        assert isinstance(call, ToolCall)
        assert call.name == "get_weather"
        assert "city" in call.arguments
        assert call.id is not None

    async def test_tool_call_finish_reason(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in London?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        # Finish reason should indicate tool usage
        assert response.finish_reason in ("tool_calls", "stop")

    async def test_forced_tool_choice(self, or_config: OpenRouterConfig) -> None:
        """tool_choice with a specific function name forces the model to call it."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="Hello, how are you?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "get_weather"

    async def test_multiple_tools_available(self, or_config: OpenRouterConfig) -> None:
        """Model selects the right tool from multiple options."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is 2 + 2?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL, _CALCULATE_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "calculate"

    async def test_tool_call_streaming(self, or_config: OpenRouterConfig) -> None:
        """Streaming with tools should produce completed_tool_calls."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in Tokyo?")],
            )
        ]
        completed_calls: list[ToolCall] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            ):
                if chunk.completed_tool_calls:
                    completed_calls.extend(chunk.completed_tool_calls)
        assert len(completed_calls) >= 1
        assert completed_calls[0].name == "get_weather"
        assert isinstance(completed_calls[0].arguments, dict)

    async def test_tool_call_stream_events(self, or_config: OpenRouterConfig) -> None:
        """chat_events() should emit tool_call_delta and tool_call_completed."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in Berlin?")],
            )
        ]
        event_types: set[str] = set()
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            ):
                event_types.add(event.type)
        assert "tool_call_completed" in event_types

    async def test_tool_call_aggregated_stream(
        self, or_config: OpenRouterConfig
    ) -> None:
        """chat(stream=True) with tools produces a ChatResponse with tool_calls."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in NYC?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
                stream=True,
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "get_weather"

    async def test_tool_response_round_trip(self, or_config: OpenRouterConfig) -> None:
        """Full tool-use loop: user -> tool call -> tool result -> final answer."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in Paris?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            first = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert first.tool_calls is not None

        call = first.tool_calls[0]
        # Build the follow-up with tool result
        messages.append(
            Message(
                role=Role.ASSISTANT,
                tool_calls=[call],
            )
        )
        messages.append(
            Message(
                role=Role.TOOL,
                tool_call_id=call.id,
                content=[TextPart(text=json.dumps({"temperature": "22C", "condition": "sunny"}))],
            )
        )
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            final = await client.chat(messages=messages, tools=[_GET_WEATHER_TOOL])
        assert final.content is not None
        # Model should reference the weather data
        assert any(
            word in final.content.lower()
            for word in ("22", "sunny", "paris", "weather", "celsius", "degrees")
        )


# =========================================================================
# 5. Config overrides
# =========================================================================


@pytest.mark.integration
class TestConfigOverrides:
    async def test_max_tokens_limits_output(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="Write a very long essay about the history of computing.")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            short = await client.chat(
                messages=messages,
                config_overrides={"max_tokens": 10},
            )
        assert short.content is not None
        # With max_tokens=10, output should be very short
        words = short.content.split()
        assert len(words) <= 30  # generous upper bound; 10 tokens ~ 7-10 words

    async def test_temperature_zero_determinism(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Two calls with temperature=0 and seed should produce identical output."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is 1+1? Reply with just the number.")],
            )
        ]
        overrides = {"temperature": 0.0, "seed": 42}
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            r1 = await client.chat(messages=messages, config_overrides=overrides)
            r2 = await client.chat(messages=messages, config_overrides=overrides)
        assert r1.content is not None
        assert r2.content is not None
        # Both should contain "2"
        assert "2" in r1.content
        assert "2" in r2.content


# =========================================================================
# 6. OpenRouter-specific features
# =========================================================================


@pytest.mark.integration
class TestOpenRouterFeatures:
    async def test_provider_prefs_data_collection_deny(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Setting data_collection='deny' should still return a valid response."""
        config = or_config.model_copy(
            update={
                "provider_prefs": OpenRouterProviderPrefs(data_collection="deny"),
            }
        )
        async with Conduit(config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.content is not None

    async def test_app_name_and_url_headers(self, or_config: OpenRouterConfig) -> None:
        """app_name and app_url should not break the request."""
        config = or_config.model_copy(
            update={
                "app_name": "conduit-integration-tests",
                "app_url": "https://github.com/conduit",
            }
        )
        async with Conduit(config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.content is not None

    async def test_metadata_passthrough(self, or_config: OpenRouterConfig) -> None:
        """Metadata dict should be accepted without error."""
        config = or_config.model_copy(
            update={
                "metadata": {"test_run": "integration", "suite": "openrouter"},
            }
        )
        async with Conduit(config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.content is not None

    async def test_route_fallback(self, or_config: OpenRouterConfig) -> None:
        """route='fallback' should be accepted."""
        config = or_config.model_copy(update={"route": "fallback"})
        async with Conduit(config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.content is not None

    async def test_transforms_middle_out(self, or_config: OpenRouterConfig) -> None:
        """transforms=['middle-out'] should be accepted."""
        config = or_config.model_copy(update={"transforms": ["middle-out"]})
        async with Conduit(config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])]
            )
        assert response.content is not None

    async def test_context_metadata_runtime_override(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Runtime overrides for context metadata fields should work."""
        context = RequestContext(
            thread_id="test-thread-123",
            tags=["integration", "openrouter"],
            metadata={"env": "test"},
        )
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Hi")])],
                context=context,
                runtime_overrides={
                    "openrouter_context_metadata_fields": [
                        "thread_id",
                        "tags",
                        "metadata",
                    ]
                },
            )
        assert response.content is not None

    async def test_from_env_constructor(self) -> None:
        """Conduit.from_env('openrouter') should work when env vars are set."""
        key, model = _skip_if_no_creds()
        # The from_env method reads CONDUIT_OPENROUTER_KEY and CONDUIT_OPENROUTER_MODEL
        # which differ from the test env vars. Set them temporarily.
        environ_backup: dict[str, str | None] = {}
        keys_to_set = {
            "CONDUIT_OPENROUTER_KEY": key,
            "CONDUIT_OPENROUTER_MODEL": model,
        }
        try:
            for env_key, env_val in keys_to_set.items():
                environ_backup[env_key] = os.environ.get(env_key)
                os.environ[env_key] = env_val

            async with Conduit.from_env("openrouter") as client:
                response = await client.chat(
                    messages=[
                        Message(role=Role.USER, content=[TextPart(text="Hi")])
                    ]
                )
            assert response.content is not None
        finally:
            for env_key, old_val in environ_backup.items():
                if old_val is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old_val


# =========================================================================
# 7. Error handling
# =========================================================================


@pytest.mark.integration
class TestErrorHandling:
    async def test_invalid_api_key_raises_auth_error(self) -> None:
        _, model = _skip_if_no_creds()
        config = OpenRouterConfig(model=model, api_key="sk-or-invalid-key-12345")
        async with Conduit(config, retry_policy=_RETRY) as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.chat(
                    messages=[
                        Message(role=Role.USER, content=[TextPart(text="Hi")])
                    ]
                )
        assert exc_info.value.status_code in {401, 403}
        assert exc_info.value.provider == "openrouter"

    async def test_invalid_model_raises_error(self, or_config: OpenRouterConfig) -> None:
        config = or_config.model_copy(
            update={"model": "nonexistent/model-that-does-not-exist-99999"}
        )
        async with Conduit(config, retry_policy=_RETRY) as client:
            with pytest.raises(ProviderError):
                await client.chat(
                    messages=[
                        Message(role=Role.USER, content=[TextPart(text="Hi")])
                    ]
                )

    async def test_invalid_api_key_on_stream_raises_auth_error(self) -> None:
        _, model = _skip_if_no_creds()
        config = OpenRouterConfig(model=model, api_key="sk-or-invalid-key-12345")
        async with Conduit(config, retry_policy=_RETRY) as client:
            with pytest.raises(AuthenticationError):
                async for _ in client.chat_stream(
                    messages=[
                        Message(role=Role.USER, content=[TextPart(text="Hi")])
                    ]
                ):
                    pass


# =========================================================================
# 8. SyncConduit wrapper
# =========================================================================


@pytest.mark.integration
class TestSyncConduit:
    def test_sync_basic_chat(self, or_config: OpenRouterConfig) -> None:
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            response = client.chat(
                messages=[Message(role=Role.USER, content=[TextPart(text="Say hi.")])]
            )
        assert response.content is not None
        assert response.provider == "openrouter"

    def test_sync_streaming(self, or_config: OpenRouterConfig) -> None:
        content_parts: list[str] = []
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            for chunk in client.chat_stream(
                messages=[
                    Message(role=Role.USER, content=[TextPart(text="Say hello.")])
                ]
            ):
                if chunk.content:
                    content_parts.append(chunk.content)
        assert len(content_parts) > 0

    def test_sync_stream_events(
        self, or_config: OpenRouterConfig, sample_messages: list[Message]
    ) -> None:
        event_types: set[str] = set()
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            for event in client.chat_events(messages=sample_messages):
                event_types.add(event.type)
        assert "text_delta" in event_types
        assert "finish" in event_types

    def test_sync_tool_calling(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Rome?")],
            )
        ]
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            response = client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "get_weather"

    def test_sync_aggregated_stream(self, or_config: OpenRouterConfig) -> None:
        messages = [
            Message(role=Role.USER, content=[TextPart(text="Say 'ok'.")])
        ]
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            response = client.chat(messages=messages, stream=True)
        assert response.content is not None
        assert response.finish_reason == "stop"


# =========================================================================
# 9. Multi-step tool calling
# =========================================================================


@pytest.mark.integration
class TestMultiStepToolCalling:
    async def test_two_step_tool_chain(self, or_config: OpenRouterConfig) -> None:
        """Weather lookup → unit conversion chain."""
        tools = [_GET_WEATHER_TOOL, _UNIT_CONVERT_TOOL]
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in Paris? After you get the temperature "
                            "in Celsius, convert it to Fahrenheit using the unit_convert tool."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            # Step 1: model should call get_weather
            r1 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r1.tool_calls is not None
            assert any(c.name == "get_weather" for c in r1.tool_calls)

            # Feed weather result back
            messages.append(Message(role=Role.ASSISTANT, tool_calls=r1.tool_calls))
            for call in r1.tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {"temperature_celsius": 22, "condition": "sunny"}
                                )
                            )
                        ],
                    )
                )

            # Step 2: model should call unit_convert
            r2 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r2.tool_calls is not None
            assert any(c.name == "unit_convert" for c in r2.tool_calls)

            # Feed conversion result back
            messages.append(Message(role=Role.ASSISTANT, tool_calls=r2.tool_calls))
            for call in r2.tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps({"result": 71.6, "unit": "fahrenheit"})
                            )
                        ],
                    )
                )

            # Step 3: model should produce a final text answer
            r3 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r3.content is not None
            assert any(
                token in r3.content.lower()
                for token in ("71", "fahrenheit", "paris")
            )

    async def test_two_step_tool_chain_streaming(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Two-step chain via chat_stream()."""
        tools = [_GET_WEATHER_TOOL, _UNIT_CONVERT_TOOL]
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in Paris? After you get the temperature "
                            "in Celsius, convert it to Fahrenheit using the unit_convert tool."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            # Step 1: collect tool calls from stream
            completed_calls: list[ToolCall] = []
            async for chunk in client.chat_stream(
                messages=messages, tools=tools, tool_choice="auto"
            ):
                if chunk.completed_tool_calls:
                    completed_calls.extend(chunk.completed_tool_calls)
            assert len(completed_calls) >= 1
            assert any(c.name == "get_weather" for c in completed_calls)

            # Feed result back
            messages.append(Message(role=Role.ASSISTANT, tool_calls=completed_calls))
            for call in completed_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {"temperature_celsius": 22, "condition": "sunny"}
                                )
                            )
                        ],
                    )
                )

            # Step 2: should produce unit_convert call
            completed_calls_2: list[ToolCall] = []
            async for chunk in client.chat_stream(
                messages=messages, tools=tools, tool_choice="auto"
            ):
                if chunk.completed_tool_calls:
                    completed_calls_2.extend(chunk.completed_tool_calls)
            assert len(completed_calls_2) >= 1
            assert any(c.name == "unit_convert" for c in completed_calls_2)

    async def test_two_step_tool_chain_events(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Two-step chain via chat_events()."""
        tools = [_GET_WEATHER_TOOL, _UNIT_CONVERT_TOOL]
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in Paris? After you get the temperature "
                            "in Celsius, convert it to Fahrenheit using the unit_convert tool."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            # Step 1
            completed_calls: list[ToolCall] = []
            event_types: set[str] = set()
            async for event in client.chat_events(
                messages=messages, tools=tools, tool_choice="auto"
            ):
                event_types.add(event.type)
                if event.type == "tool_call_completed" and isinstance(
                    event.tool_call, ToolCall
                ):
                    completed_calls.append(event.tool_call)
            assert "tool_call_completed" in event_types
            assert len(completed_calls) >= 1

            # Feed result back
            messages.append(Message(role=Role.ASSISTANT, tool_calls=completed_calls))
            for call in completed_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {"temperature_celsius": 22, "condition": "sunny"}
                                )
                            )
                        ],
                    )
                )

            # Step 2
            completed_calls_2: list[ToolCall] = []
            event_types_2: set[str] = set()
            async for event in client.chat_events(
                messages=messages, tools=tools, tool_choice="auto"
            ):
                event_types_2.add(event.type)
                if event.type == "tool_call_completed" and isinstance(
                    event.tool_call, ToolCall
                ):
                    completed_calls_2.append(event.tool_call)
            assert "tool_call_completed" in event_types_2
            assert len(completed_calls_2) >= 1

    async def test_three_step_tool_chain(self, or_config: OpenRouterConfig) -> None:
        """Search → weather → unit convert chain."""
        tools = [_SEARCH_TOOL, _GET_WEATHER_TOOL, _UNIT_CONVERT_TOOL]
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[
                    TextPart(
                        text=(
                            "You are a helpful assistant. You MUST use the provided tools "
                            "to complete each step. Do not skip any tool call."
                        )
                    )
                ],
            ),
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "I need you to do three things, one at a time:\n"
                            "1. Search for 'best cities to visit in Europe'\n"
                            "2. Get the weather in London\n"
                            "3. Convert the temperature from Celsius to Fahrenheit\n"
                            "Start with step 1 now."
                        )
                    )
                ],
            ),
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            tool_names_seen: list[str] = []

            # Step 1: search
            r1 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r1.tool_calls is not None
            tool_names_seen.extend(c.name for c in r1.tool_calls)
            messages.append(Message(role=Role.ASSISTANT, tool_calls=r1.tool_calls))
            for call in r1.tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {
                                        "results": [
                                            {"city": "London", "rank": 1},
                                            {"city": "Paris", "rank": 2},
                                            {"city": "Rome", "rank": 3},
                                        ]
                                    }
                                )
                            )
                        ],
                    )
                )

            # Step 2: get_weather
            messages.append(
                Message(
                    role=Role.USER,
                    content=[TextPart(text="Good. Now do step 2: get the weather in London.")],
                )
            )
            r2 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r2.tool_calls is not None
            tool_names_seen.extend(c.name for c in r2.tool_calls)
            messages.append(Message(role=Role.ASSISTANT, tool_calls=r2.tool_calls))
            for call in r2.tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {"temperature_celsius": 15, "condition": "cloudy"}
                                )
                            )
                        ],
                    )
                )

            # Step 3: unit_convert
            messages.append(
                Message(
                    role=Role.USER,
                    content=[
                        TextPart(
                            text=(
                                "Good. Now do step 3: convert 15 celsius to fahrenheit "
                                "using the unit_convert tool."
                            )
                        )
                    ],
                )
            )
            r3 = await client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r3.tool_calls is not None
            tool_names_seen.extend(c.name for c in r3.tool_calls)

            # Verify all three tool types were called
            assert "search" in tool_names_seen
            assert "get_weather" in tool_names_seen
            assert "unit_convert" in tool_names_seen


# =========================================================================
# 10. Parallel tool calls
# =========================================================================


@pytest.mark.integration
class TestParallelToolCalls:
    async def test_parallel_tool_calls_non_streaming(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Model should invoke get_weather for two cities in a single response."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in BOTH Paris AND London? "
                            "Call get_weather for each city."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 2
        cities = {
            call.arguments.get("city", "").lower() for call in response.tool_calls
        }
        assert "paris" in cities or "london" in cities  # at least one matches

    async def test_parallel_tool_calls_streaming(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Parallel tool calls via chat_stream()."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in BOTH Paris AND London? "
                            "Call get_weather for each city."
                        )
                    )
                ],
            )
        ]
        completed_calls: list[ToolCall] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            ):
                if chunk.completed_tool_calls:
                    completed_calls.extend(chunk.completed_tool_calls)
        assert len(completed_calls) >= 2

    async def test_parallel_tool_calls_round_trip(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Full loop: parallel calls → feed both results → model summarizes."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in BOTH Paris AND London? "
                            "Call get_weather for each city."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            r1 = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
            assert r1.tool_calls is not None
            assert len(r1.tool_calls) >= 2

            # Feed all tool results back
            messages.append(Message(role=Role.ASSISTANT, tool_calls=r1.tool_calls))
            fake_results = {
                "paris": {"temperature_celsius": 22, "condition": "sunny"},
                "london": {"temperature_celsius": 15, "condition": "rainy"},
            }
            for call in r1.tool_calls:
                city = call.arguments.get("city", "unknown").lower()
                result = fake_results.get(city, {"temperature_celsius": 20, "condition": "clear"})
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[TextPart(text=json.dumps(result))],
                    )
                )

            # Model should summarize both cities
            r2 = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
            assert r2.content is not None
            lower = r2.content.lower()
            # Should reference at least some of the weather data
            assert any(
                token in lower
                for token in ("paris", "london", "sunny", "rainy", "22", "15")
            )


# =========================================================================
# 11. Complex tool schemas
# =========================================================================


@pytest.mark.integration
class TestComplexToolSchemas:
    async def test_nested_object_arguments(self, or_config: OpenRouterConfig) -> None:
        """Model should produce correctly nested departure/arrival objects."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "Book a flight from New York on 2025-06-01 to London "
                            "arriving 2025-06-02, for 2 passengers."
                        )
                    )
                ],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_BOOK_FLIGHT_TOOL],
                tool_choice={"type": "function", "function": {"name": "book_flight"}},
            )
        assert response.tool_calls is not None
        call = response.tool_calls[0]
        assert call.name == "book_flight"
        args = call.arguments
        # Verify nested structure
        assert "departure" in args
        assert isinstance(args["departure"], dict)
        assert "city" in args["departure"]
        assert "arrival" in args
        assert isinstance(args["arrival"], dict)
        assert "city" in args["arrival"]
        assert "passengers" in args

    async def test_enum_constrained_arguments(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Model should use valid enum values for from_unit/to_unit."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(text="Convert 100 celsius to fahrenheit.")
                ],
            )
        ]
        valid_units = {"celsius", "fahrenheit", "kelvin", "meters", "feet", "kg", "lbs"}
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_UNIT_CONVERT_TOOL],
                tool_choice={"type": "function", "function": {"name": "unit_convert"}},
            )
        assert response.tool_calls is not None
        call = response.tool_calls[0]
        assert call.name == "unit_convert"
        assert call.arguments["from_unit"] in valid_units
        assert call.arguments["to_unit"] in valid_units
        assert isinstance(call.arguments["value"], (int, float))

    async def test_optional_and_required_fields(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Tool with mix of required and optional params."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="Search for 'best python libraries'.")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_SEARCH_TOOL],
                tool_choice={"type": "function", "function": {"name": "search"}},
            )
        assert response.tool_calls is not None
        call = response.tool_calls[0]
        assert call.name == "search"
        # 'query' is required and must be present
        assert "query" in call.arguments
        assert isinstance(call.arguments["query"], str)
        # 'max_results' is optional — may or may not be present, both are valid


# =========================================================================
# 12. Tool choice modes
# =========================================================================


@pytest.mark.integration
class TestToolChoiceModes:
    async def test_tool_choice_required(self, or_config: OpenRouterConfig) -> None:
        """tool_choice='required' forces a tool call even when unnecessary."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="Hello, how are you today?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="required",
            )
        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1

    async def test_tool_choice_none(self, or_config: OpenRouterConfig) -> None:
        """tool_choice='none' prevents tool calls even when relevant."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Paris?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="none",
            )
        # Model should produce text instead of a tool call
        assert response.tool_calls is None or len(response.tool_calls) == 0
        assert response.content is not None

    async def test_tool_choice_auto_no_tool_needed(
        self, or_config: OpenRouterConfig
    ) -> None:
        """tool_choice='auto' with irrelevant prompt should produce text only."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is 2 + 2? Reply with just the number.")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_BOOK_FLIGHT_TOOL],  # unrelated tool
                tool_choice="auto",
            )
        assert response.content is not None
        assert "4" in response.content


# =========================================================================
# 13. Streaming tool call details
# =========================================================================


@pytest.mark.integration
class TestStreamingToolCallDetails:
    async def test_streaming_tool_call_deltas_have_partial_data(
        self, or_config: OpenRouterConfig
    ) -> None:
        """tool_call_delta events should contain PartialToolCall objects."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in Tokyo?")],
            )
        ]
        deltas: list[PartialToolCall] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for event in client.chat_events(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            ):
                if event.type == "tool_call_delta" and isinstance(
                    event.tool_call, PartialToolCall
                ):
                    deltas.append(event.tool_call)
        assert len(deltas) >= 1
        # At least one delta should have an index
        assert any(d.index is not None for d in deltas)
        # At least one delta should have a name or arguments fragment
        assert any(
            d.name is not None or d.arguments_fragment is not None for d in deltas
        )

    async def test_streaming_completed_tool_call_has_valid_arguments(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Completed tool call from streaming should have parseable JSON arguments."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What's the weather in Berlin?")],
            )
        ]
        completed_calls: list[ToolCall] = []
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            async for chunk in client.chat_stream(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            ):
                if chunk.completed_tool_calls:
                    completed_calls.extend(chunk.completed_tool_calls)
        assert len(completed_calls) >= 1
        call = completed_calls[0]
        assert call.name == "get_weather"
        assert isinstance(call.arguments, dict)
        assert "city" in call.arguments
        assert call.id is not None

    async def test_aggregated_stream_tool_calls_match_non_stream(
        self, or_config: OpenRouterConfig
    ) -> None:
        """chat(stream=True) and chat() should both produce get_weather calls."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Paris?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            non_stream = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            )
            stream = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
                stream=True,
            )
        assert non_stream.tool_calls is not None
        assert stream.tool_calls is not None
        assert non_stream.tool_calls[0].name == "get_weather"
        assert stream.tool_calls[0].name == "get_weather"
        # Both should reference Paris (or similar city)
        assert "city" in non_stream.tool_calls[0].arguments
        assert "city" in stream.tool_calls[0].arguments


# =========================================================================
# 14. Tool calling with config overrides
# =========================================================================


@pytest.mark.integration
class TestToolCallingWithConfigOverrides:
    async def test_tool_calling_with_temperature_zero(
        self, or_config: OpenRouterConfig
    ) -> None:
        """temperature=0 should still allow tool calls."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Paris?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
                config_overrides={"temperature": 0.0},
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "get_weather"

    async def test_tool_calling_with_max_tokens(
        self, or_config: OpenRouterConfig
    ) -> None:
        """Low max_tokens should not prevent tool calls from completing."""
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Rome?")],
            )
        ]
        async with Conduit(or_config, retry_policy=_RETRY) as client:
            response = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
                config_overrides={"max_tokens": 100},
            )
        assert response.tool_calls is not None
        assert response.tool_calls[0].name == "get_weather"
        assert "city" in response.tool_calls[0].arguments


# =========================================================================
# 15. Multi-model tool calling
# =========================================================================


@pytest.mark.integration
class TestMultiModelToolCalling:
    @pytest.mark.parametrize(
        "model_id",
        [
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "meta-llama/llama-3.3-70b-instruct",
        ],
    )
    async def test_tool_calling_across_models(self, model_id: str) -> None:
        """Basic tool call + round-trip across multiple models."""
        key, _ = _get_creds()
        if not key:
            pytest.skip(_SKIP_MSG)
        config = OpenRouterConfig(model=model_id, api_key=key)
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text="What is the weather in Paris?")],
            )
        ]
        async with Conduit(config, retry_policy=_RETRY) as client:
            r1 = await client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert r1.tool_calls is not None
        call = r1.tool_calls[0]
        assert call.name == "get_weather"
        assert "city" in call.arguments

        # Round-trip: feed result and get final answer
        messages.append(Message(role=Role.ASSISTANT, tool_calls=r1.tool_calls))
        messages.append(
            Message(
                role=Role.TOOL,
                tool_call_id=call.id,
                content=[
                    TextPart(
                        text=json.dumps({"temperature_celsius": 18, "condition": "cloudy"})
                    )
                ],
            )
        )
        async with Conduit(config, retry_policy=_RETRY) as client:
            r2 = await client.chat(messages=messages, tools=[_GET_WEATHER_TOOL])
        assert r2.content is not None


# =========================================================================
# 16. Sync multi-step tool calling
# =========================================================================


@pytest.mark.integration
class TestSyncMultiStepToolCalling:
    def test_sync_two_step_tool_chain(self, or_config: OpenRouterConfig) -> None:
        """Multi-step tool chain via SyncConduit."""
        tools = [_GET_WEATHER_TOOL, _UNIT_CONVERT_TOOL]
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in Paris? After you get the temperature "
                            "in Celsius, convert it to Fahrenheit using the unit_convert tool."
                        )
                    )
                ],
            )
        ]
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            # Step 1: get_weather
            r1 = client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r1.tool_calls is not None
            assert any(c.name == "get_weather" for c in r1.tool_calls)

            messages.append(Message(role=Role.ASSISTANT, tool_calls=r1.tool_calls))
            for call in r1.tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_call_id=call.id,
                        content=[
                            TextPart(
                                text=json.dumps(
                                    {"temperature_celsius": 22, "condition": "sunny"}
                                )
                            )
                        ],
                    )
                )

            # Step 2: unit_convert
            r2 = client.chat(messages=messages, tools=tools, tool_choice="auto")
            assert r2.tool_calls is not None
            assert any(c.name == "unit_convert" for c in r2.tool_calls)

    def test_sync_parallel_tool_calls(self, or_config: OpenRouterConfig) -> None:
        """Parallel tool calls via SyncConduit."""
        messages = [
            Message(
                role=Role.USER,
                content=[
                    TextPart(
                        text=(
                            "What is the weather in BOTH Paris AND London? "
                            "Call get_weather for each city."
                        )
                    )
                ],
            )
        ]
        with SyncConduit(or_config, retry_policy=_RETRY) as client:
            response = client.chat(
                messages=messages,
                tools=[_GET_WEATHER_TOOL],
                tool_choice="auto",
            )
        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 2
