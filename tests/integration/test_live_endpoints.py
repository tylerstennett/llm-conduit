from __future__ import annotations

import os

import pytest

from conduit import Conduit, Message, OllamaConfig, OpenRouterConfig, Role, VLLMConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vllm_live_basic_chat() -> None:
    url = os.getenv("CONDUIT_TEST_VLLM_URL")
    model = os.getenv("CONDUIT_TEST_VLLM_MODEL")
    if not url or not model:
        pytest.skip("set CONDUIT_TEST_VLLM_URL and CONDUIT_TEST_VLLM_MODEL to run")

    config = VLLMConfig(model=model, base_url=url)
    async with Conduit(config) as client:
        response = await client.chat(messages=[Message(role=Role.USER, content="Say hello")])
    assert response.content is not None or response.tool_calls is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_live_basic_chat() -> None:
    url = os.getenv("CONDUIT_TEST_OLLAMA_URL")
    model = os.getenv("CONDUIT_TEST_OLLAMA_MODEL")
    if not url or not model:
        pytest.skip("set CONDUIT_TEST_OLLAMA_URL and CONDUIT_TEST_OLLAMA_MODEL to run")

    config = OllamaConfig(model=model, base_url=url)
    async with Conduit(config) as client:
        response = await client.chat(messages=[Message(role=Role.USER, content="Say hello")])
    assert response.content is not None or response.tool_calls is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openrouter_live_basic_chat() -> None:
    model = os.getenv("CONDUIT_TEST_OPENROUTER_MODEL")
    api_key = os.getenv("CONDUIT_TEST_OPENROUTER_KEY")
    if not model or not api_key:
        pytest.skip(
            "set CONDUIT_TEST_OPENROUTER_MODEL and CONDUIT_TEST_OPENROUTER_KEY to run"
        )

    config = OpenRouterConfig(model=model, api_key=api_key)
    async with Conduit(config) as client:
        response = await client.chat(messages=[Message(role=Role.USER, content="Say hello")])
    assert response.content is not None or response.tool_calls is not None

