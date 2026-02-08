from __future__ import annotations

import asyncio

import pytest

from conduit.client import Conduit
from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.providers import OllamaProvider, OpenRouterProvider, VLLMProvider


def test_create_provider_uses_registry_for_config_types() -> None:
    clients = [
        Conduit(VLLMConfig(model="m")),
        Conduit(OllamaConfig(model="m")),
        Conduit(OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k")),
    ]
    try:
        assert isinstance(clients[0]._provider, VLLMProvider)
        assert isinstance(clients[1]._provider, OllamaProvider)
        assert isinstance(clients[2]._provider, OpenRouterProvider)
    finally:
        for client in clients:
            asyncio.run(client.aclose())


def test_from_env_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDUIT_OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("CONDUIT_OPENROUTER_KEY", "k")

    client = Conduit.from_env("openrouter")
    try:
        assert isinstance(client._provider, OpenRouterProvider)
        assert isinstance(client.config, OpenRouterConfig)
    finally:
        asyncio.run(client.aclose())


def test_from_env_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigValidationError, match="provider must be one of"):
        Conduit.from_env("does-not-exist")
