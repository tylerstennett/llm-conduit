from __future__ import annotations

import pytest

from conduit.config import OllamaConfig, OpenRouterConfig, OpenRouterProviderPrefs, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.models.messages import ChatRequest, Message, Role
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openrouter import OpenRouterProvider
from conduit.providers.vllm import VLLMProvider


def test_vllm_request_builds_structured_outputs_from_guided_alias(
    sample_messages,
    sample_tools,
) -> None:
    config = VLLMConfig(
        model="m",
        guided_regex="[0-9]+",
        top_k=50,
        response_format={"type": "json_object"},
    )
    provider = VLLMProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=config,
        stream=False,
    )

    assert body["structured_outputs"]["regex"] == "[0-9]+"
    assert body["top_k"] == 50
    assert body["response_format"] == {"type": "json_object"}
    assert body["tools"][0]["type"] == "function"


def test_vllm_stream_request_includes_stream_options(sample_messages) -> None:
    config = VLLMConfig(model="m", stream_options={"include_usage": True})
    provider = VLLMProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, stream=True),
        effective_config=config,
        stream=True,
    )

    assert body["stream_options"] == {"include_usage": True}


def test_vllm_rejects_stream_options_in_non_stream_mode(sample_messages) -> None:
    config = VLLMConfig(model="m", stream_options={"include_usage": True})
    provider = VLLMProvider(config)

    with pytest.raises(ConfigValidationError):
        provider.build_request_body(
            ChatRequest(messages=sample_messages),
            effective_config=config,
            stream=False,
        )


def test_ollama_request_places_generation_params_in_options(
    sample_messages,
    sample_tools,
) -> None:
    config = OllamaConfig(model="m", temperature=0.2, max_tokens=77, top_k=10)
    provider = OllamaProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=config,
        stream=False,
    )

    assert "options" in body
    assert body["options"]["temperature"] == 0.2
    assert body["options"]["num_predict"] == 77
    assert body["tools"][0]["type"] == "function"


def test_ollama_tool_choice_none_omits_tools(sample_messages, sample_tools) -> None:
    config = OllamaConfig(model="m")
    provider = OllamaProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="none"),
        effective_config=config,
        stream=False,
    )

    assert "tools" not in body


def test_ollama_generate_mode_builds_prompt_request(sample_messages) -> None:
    config = OllamaConfig(model="m", raw=True, suffix=" END")
    provider = OllamaProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages),
        effective_config=config,
        stream=False,
    )

    assert body["prompt"] == "What is the weather in New York?"
    assert body["system"] == "You are helpful."
    assert body["raw"] is True
    assert body["suffix"] == " END"
    assert "messages" not in body


def test_ollama_generate_mode_rejects_tools(sample_messages, sample_tools) -> None:
    config = OllamaConfig(model="m", raw=True)
    provider = OllamaProvider(config)

    with pytest.raises(ConfigValidationError):
        provider.build_request_body(
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
            effective_config=config,
            stream=False,
        )


def test_ollama_generate_mode_rejects_multiple_user_messages() -> None:
    config = OllamaConfig(model="m", raw=True)
    provider = OllamaProvider(config)
    messages = [
        Message(role=Role.USER, content="A"),
        Message(role=Role.USER, content="B"),
    ]

    with pytest.raises(ConfigValidationError):
        provider.build_request_body(
            ChatRequest(messages=messages),
            effective_config=config,
            stream=False,
        )


def test_ollama_rejects_required_or_named_tool_choice(sample_messages, sample_tools) -> None:
    config = OllamaConfig(model="m")
    provider = OllamaProvider(config)

    with pytest.raises(ConfigValidationError):
        provider.build_request_body(
            ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="required"),
            effective_config=config,
            stream=False,
        )


def test_openrouter_request_includes_provider_route_and_headers(
    sample_messages,
    sample_tools,
) -> None:
    config = OpenRouterConfig(
        model="openai/gpt-4o-mini",
        api_key="secret",
        route="fallback",
        provider_prefs=OpenRouterProviderPrefs(order=["together"]),
        app_name="Conduit",
        app_url="https://example.com",
    )
    provider = OpenRouterProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=config,
        stream=False,
    )
    headers = provider.default_headers()

    assert body["route"] == "fallback"
    assert body["provider"]["order"] == ["together"]
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "Conduit"


def test_openrouter_request_includes_reasoning_transforms_and_include(
    sample_messages,
) -> None:
    config = OpenRouterConfig(
        model="openai/gpt-4o-mini",
        api_key="secret",
        reasoning={"effort": "medium"},
        transforms=["middle-out"],
        include=["reasoning"],
    )
    provider = OpenRouterProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages),
        effective_config=config,
        stream=False,
    )

    assert body["reasoning"] == {"effort": "medium"}
    assert body["transforms"] == ["middle-out"]
    assert body["include"] == ["reasoning"]
