from __future__ import annotations

import httpx
import pytest

from conduit.client import Conduit
from conduit.config import OpenRouterConfig, OpenRouterProviderPrefs, VLLMConfig
from conduit.exceptions import ConfigValidationError
from conduit.models.messages import Message, Role


def test_overrides_apply_and_revalidate() -> None:
    client = Conduit(VLLMConfig(model="m", temperature=0.7))
    updated = client._apply_overrides({"temperature": 0.1})
    assert isinstance(updated, VLLMConfig)
    assert updated.temperature == 0.1


@pytest.mark.asyncio
async def test_unknown_override_field_raises() -> None:
    client = Conduit(VLLMConfig(model="m"))
    with pytest.raises(ConfigValidationError):
        await client.chat(messages=[], config_overrides={"unknown": 1})


def test_openrouter_nested_override_merges_provider_object() -> None:
    client = Conduit(
        OpenRouterConfig(
            model="openai/gpt-4o-mini",
            api_key="k",
            provider_prefs=OpenRouterProviderPrefs(order=["a"]),
        )
    )
    updated = client._apply_overrides(
        {
            "provider": {
                "order": ["b", "c"],
                "allow_fallbacks": False,
            }
        }
    )
    assert isinstance(updated, OpenRouterConfig)
    assert updated.provider_prefs is not None
    assert updated.provider_prefs.order == ["b", "c"]
    assert updated.provider_prefs.allow_fallbacks is False


def test_vllm_new_fields_accept_overrides() -> None:
    client = Conduit(VLLMConfig(model="m"))
    updated = client._apply_overrides(
        {
            "response_format": {"type": "json_object"},
            "stream_options": {"include_usage": True},
        }
    )
    assert isinstance(updated, VLLMConfig)
    assert updated.response_format == {"type": "json_object"}
    assert updated.stream_options == {"include_usage": True}


def test_openrouter_new_fields_accept_overrides() -> None:
    client = Conduit(OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"))
    updated = client._apply_overrides(
        {
            "reasoning": {"effort": "high"},
            "transforms": ["middle-out"],
            "include": ["reasoning"],
        }
    )
    assert isinstance(updated, OpenRouterConfig)
    assert updated.reasoning == {"effort": "high"}
    assert updated.transforms == ["middle-out"]
    assert updated.include == ["reasoning"]


@pytest.mark.asyncio
async def test_transport_uses_override_base_url_and_api_key() -> None:
    requested_url: str | None = None
    requested_auth: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_url, requested_auth
        requested_url = str(request.url)
        requested_auth = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    client = Conduit(VLLMConfig(model="m", base_url="http://old/v1", api_key="old"))
    client._provider.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._provider._client_owned = True

    await client.chat(
        messages=[Message(role=Role.USER, content="hello")],
        config_overrides={"base_url": "http://new/v1", "api_key": "new"},
    )
    await client.aclose()

    assert requested_url == "http://new/v1/chat/completions"
    assert requested_auth == "Bearer new"


@pytest.mark.asyncio
async def test_openrouter_headers_use_override_values() -> None:
    referer: str | None = None
    title: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal referer, title
        referer = request.headers.get("HTTP-Referer")
        title = request.headers.get("X-Title")
        return httpx.Response(
            200,
            json={
                "id": "or-1",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    client = Conduit(
        OpenRouterConfig(
            model="openai/gpt-4o-mini",
            api_key="k",
            app_name="Old App",
            app_url="https://old.example.com",
        )
    )
    client._provider.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._provider._client_owned = True

    await client.chat(
        messages=[Message(role=Role.USER, content="hello")],
        config_overrides={
            "app_name": "New App",
            "app_url": "https://new.example.com",
        },
    )
    await client.aclose()

    assert referer == "https://new.example.com"
    assert title == "New App"
