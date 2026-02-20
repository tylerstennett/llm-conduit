from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from conduit.config import OllamaConfig, OpenRouterConfig, VLLMConfig
from conduit.exceptions import (
    AuthenticationError,
    ContextLengthError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    StreamError,
    ToolCallParseError,
)
from conduit.models.messages import ChatRequest, Message, Role, TextPart
from conduit.providers import BaseProvider
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openrouter import OpenRouterProvider
from conduit.providers.vllm import VLLMProvider


class _SingleBodyStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


@pytest.mark.asyncio
async def test_http_status_error_mapping() -> None:
    status_payloads = [
        (401, {"error": {"message": "bad auth"}}, AuthenticationError),
        (429, {"error": {"message": "too many"}}, RateLimitError),
        (404, {"error": {"message": "model not found"}}, ModelNotFoundError),
        (400, {"error": {"message": "context length exceeded"}}, ContextLengthError),
        (500, {"error": {"message": "server down"}}, ProviderUnavailableError),
    ]

    for status_code, payload, expected_type in status_payloads:
        transport = httpx.MockTransport(
            lambda request, status_code=status_code, payload=payload: httpx.Response(
                status_code,
                json=payload,
            )
        )
        client = httpx.AsyncClient(transport=transport)
        provider = VLLMProvider(VLLMConfig(model="m"), http_client=client)

        with pytest.raises(expected_type):
            await provider.chat(
                ChatRequest(
                    messages=[Message(role=Role.USER, content=[TextPart(text="hi")])]
                ),
                effective_config=provider.config,
            )

        await client.aclose()


@pytest.mark.asyncio
async def test_malformed_sse_raises_stream_error() -> None:
    bad_sse = "data: {not-json}\n\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=bad_sse, headers={"content-type": "text/event-stream"})
    )
    client = httpx.AsyncClient(transport=transport)
    provider = VLLMProvider(VLLMConfig(model="m"), http_client=client)

    with pytest.raises(StreamError):
        async for _ in provider.chat_stream(
            ChatRequest(
                messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
                stream=True,
            ),
            effective_config=provider.config,
        ):
            pass

    await client.aclose()


def test_ollama_missing_tool_name_mapping_raises() -> None:
    provider = OllamaProvider(OllamaConfig(model="m"))

    with pytest.raises(ToolCallParseError):
        provider.build_request_body(
            ChatRequest(
                messages=[
                    Message(
                        role=Role.TOOL,
                        tool_call_id="missing",
                        content=[TextPart(text="{}")],
                    ),
                ]
            ),
            effective_config=provider.config,
            stream=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda client: VLLMProvider(VLLMConfig(model="m"), http_client=client),
        lambda client: OllamaProvider(OllamaConfig(model="m"), http_client=client),
        lambda client: OpenRouterProvider(
            OpenRouterConfig(model="openai/gpt-4o-mini", api_key="k"),
            http_client=client,
        ),
    ],
)
async def test_stream_http_errors_raise_provider_errors_not_response_not_read(
    provider_factory: Callable[[httpx.AsyncClient], BaseProvider],
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            429,
            json={"error": {"message": "too many requests"}},
        )
    )
    client = httpx.AsyncClient(transport=transport)
    provider = provider_factory(client)

    with pytest.raises(RateLimitError, match="too many requests"):
        async for _ in provider.chat_stream(
            ChatRequest(
                messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
                stream=True,
            ),
            effective_config=provider.config,
        ):
            pass

    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_generate_stream_http_error_reads_stream_before_mapping() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            429,
            stream=_SingleBodyStream(b'{"error":{"message":"too many requests"}}'),
        )
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OllamaProvider(OllamaConfig(model="m", raw=True), http_client=client)

    with pytest.raises(RateLimitError, match="too many requests"):
        async for _ in provider.chat_stream(
            ChatRequest(
                messages=[Message(role=Role.USER, content=[TextPart(text="hi")])],
                stream=True,
            ),
            effective_config=provider.config,
        ):
            pass

    await client.aclose()
