from __future__ import annotations

import pytest

from conduit.config import OllamaConfig, OpenRouterConfig, OpenRouterProviderPrefs, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.models.messages import ChatRequest
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openrouter import OpenRouterProvider
from conduit.providers.vllm import VLLMProvider


def test_vllm_request_builds_structured_outputs_from_guided_alias(
    sample_messages,
    sample_tools,
) -> None:
    config = VLLMConfig(model="m", guided_regex="[0-9]+", top_k=50)
    provider = VLLMProvider(config)

    body = provider.build_request_body(
        ChatRequest(messages=sample_messages, tools=sample_tools, tool_choice="auto"),
        effective_config=config,
        stream=False,
    )

    assert body["structured_outputs"]["regex"] == "[0-9]+"
    assert body["top_k"] == 50
    assert body["tools"][0]["type"] == "function"


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

