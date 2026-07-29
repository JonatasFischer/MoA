from __future__ import annotations

import json

import httpx
import pytest

from moa_gateway.app import create_app
from moa_gateway.config import GatewayConfig


@pytest.mark.asyncio
async def test_health_and_model_discovery(api) -> None:
    client, _ = api

    health = await client.get("/health")
    models = await client.get("/v1/models?limit=1000")

    assert health.json() == {"status": "ok"}
    assert [item["id"] for item in models.json()["data"]] == [
        "claude-moa-code",
        "moa-code",
    ]


@pytest.mark.asyncio
async def test_chat_completion_translation(api) -> None:
    client, provider = api

    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 100,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello from local"
    assert response.json()["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    model, request = provider.requests[0]
    assert model == "local-model"
    assert request.messages == [{"role": "user", "content": "hi"}]
    assert request.max_tokens == 100


@pytest.mark.asyncio
async def test_anthropic_message_translation(api) -> None:
    client, provider = api

    response = await client.post(
        "/v1/messages?beta=true",
        headers={"anthropic-version": "2023-06-01"},
        json={
            "model": "claude-moa-code",
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    assert response.json()["type"] == "message"
    assert response.json()["content"] == [
        {"type": "text", "text": "hello from local"}
    ]
    assert provider.requests[0][1].messages[0] == {
        "role": "system",
        "content": "Be concise.",
    }


@pytest.mark.asyncio
async def test_responses_translation(api) -> None:
    client, provider = api

    response = await client.post(
        "/v1/responses",
        json={
            "model": "moa-code",
            "instructions": "Be concise.",
            "input": "hi",
            "max_output_tokens": 100,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "hello from local"
    assert provider.requests[0][1].messages == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "moa-code",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "read"}}],
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-moa-code",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            },
        ),
        (
            "/v1/responses",
            {
                "model": "moa-code",
                "input": "hi",
                "tools": [{"type": "function", "name": "read"}],
            },
        ),
    ],
)
async def test_tools_are_rejected_instead_of_dropped(api, path, body) -> None:
    client, provider = api

    response = await client.post(path, json=body)

    assert response.status_code == 501
    assert "silently discarded" in response.json()["detail"]
    assert provider.requests == []


@pytest.mark.asyncio
async def test_unknown_model_returns_404(api) -> None:
    client, _ = api
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "missing",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_bearer_and_api_key_auth(gateway_config, api, monkeypatch) -> None:
    monkeypatch.setenv("TEST_MOA_KEY", "secret")
    secured = gateway_config.model_copy(deep=True)
    secured.server.api_key_env = "TEST_MOA_KEY"
    _, provider = api
    app = create_app(secured, {"local": provider})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/v1/models")).status_code == 401
        assert (
            await client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            )
        ).status_code == 200
        assert (
            await client.get("/v1/models", headers={"x-api-key": "secret"})
        ).status_code == 200


def _event_types(text: str) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        if line.startswith("data: {"):
            result.append(json.loads(line[6:])["type"])
    return result


@pytest.mark.asyncio
async def test_all_streaming_protocols(api) -> None:
    client, _ = api

    chat = await client.post(
        "/v1/chat/completions",
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    anthropic = await client.post(
        "/v1/messages",
        json={
            "model": "claude-moa-code",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20,
            "stream": True,
        },
    )
    responses = await client.post(
        "/v1/responses",
        json={"model": "moa-code", "input": "hi", "stream": True},
    )

    assert "hello " in chat.text
    assert chat.text.endswith("data: [DONE]\n\n")
    assert _event_types(anthropic.text) == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert _event_types(responses.text)[-1] == "response.completed"
