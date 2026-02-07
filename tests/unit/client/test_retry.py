from __future__ import annotations

import httpx
import pytest

from conduit.client import Conduit
from conduit.config import VLLMConfig
from conduit.models.messages import Message, Role
from conduit.retry import RetryPolicy


@pytest.mark.asyncio
async def test_chat_retries_on_retryable_errors() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            json={
                "id": "1",
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

    transport = httpx.MockTransport(handler)
    config = VLLMConfig(model="m")
    conduit = Conduit(config, retry_policy=RetryPolicy(max_retries=2, backoff_base=0.0, jitter=0.0))
    conduit._provider.http_client = httpx.AsyncClient(transport=transport)
    conduit._provider._client_owned = True

    response = await conduit.chat(messages=[Message(role=Role.USER, content="hi")])

    assert response.content == "ok"
    assert call_count == 2

    await conduit.aclose()


@pytest.mark.asyncio
async def test_chat_retry_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("conduit.client.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "2.5"},
                json={"error": {"message": "rate limited"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "1",
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

    transport = httpx.MockTransport(handler)
    config = VLLMConfig(model="m")
    conduit = Conduit(config, retry_policy=RetryPolicy(max_retries=2, backoff_base=0.0, jitter=0.0))
    conduit._provider.http_client = httpx.AsyncClient(transport=transport)
    conduit._provider._client_owned = True

    response = await conduit.chat(messages=[Message(role=Role.USER, content="hi")])

    assert response.content == "ok"
    assert call_count == 2
    assert sleep_calls == [2.5]

    await conduit.aclose()
