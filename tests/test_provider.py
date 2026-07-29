from __future__ import annotations

import json

import httpx
import pytest

from moa_gateway.config import ProviderConfig
from moa_gateway.domain import CanonicalRequest
from moa_gateway.provider import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_openai_compatible_complete_and_stream() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body["stream"]:
            content = (
                'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=content, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    config = ProviderConfig.model_validate(
        {"type": "openai-compatible", "base_url": "http://ollama.test/v1"}
    )
    provider = OpenAICompatibleProvider(config)
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="http://ollama.test/v1/", transport=httpx.MockTransport(handler)
    )
    canonical = CanonicalRequest(
        requested_model="moa-code",
        messages=[{"role": "user", "content": "hi"}],
    )

    completion = await provider.complete("local-model", canonical)
    events = [event async for event in provider.stream("local-model", canonical)]
    await provider.close()

    assert completion.content == "hello"
    assert completion.usage.total_tokens == 5
    assert events[0].content == "hello"
    assert events[-1].done is True
    assert events[-1].usage.total_tokens == 5
    assert requests[0]["model"] == "local-model"
    assert requests[1]["stream_options"] == {"include_usage": True}
