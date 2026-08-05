from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from moa_gateway.app import create_app
from moa_gateway.config import GatewayConfig, load_config
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent


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
async def test_flow_lab_and_config_api_are_public(api) -> None:
    client, _ = api

    page = await client.get("/")
    script = await client.get("/assets/app.js")
    config = await client.get("/api/config")

    assert page.status_code == 200
    assert "MoA Flow Lab" in page.text
    assert "Request filter" in script.text
    assert "Tool-call gate" in script.text
    assert "No semantic refinement stage" in script.text
    assert "runSimulation" in script.text
    assert config.status_code == 200
    assert config.json()["generation"] == 1
    assert config.json()["persisted"] is False
    assert config.json()["config"]["default_profile"] == "code"
    assert config.json()["config"]["tool_enforcement"]["enabled"] is False


@pytest.mark.asyncio
async def test_config_update_applies_to_the_next_request(api) -> None:
    client, provider = api
    payload = (await client.get("/api/config")).json()["config"]
    payload["profiles"]["code"]["model"] = "experimental-model"

    updated = await client.put("/api/config", json=payload)
    completion = await client.post(
        "/v1/chat/completions",
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "test the flow"}],
        },
    )

    assert updated.status_code == 200
    assert updated.json()["generation"] == 2
    assert completion.status_code == 200
    assert provider.requests[-1][0] == "experimental-model"


@pytest.mark.asyncio
async def test_config_update_applies_tool_enforcement(api) -> None:
    client, _ = api
    payload = (await client.get("/api/config")).json()["config"]
    payload["tool_enforcement"] = {
        "enabled": True,
        "investigation_tools": ["task"],
        "max_investigation_calls": 3,
    }

    updated = await client.put("/api/config", json=payload)

    assert updated.status_code == 200
    current = updated.json()["config"]["tool_enforcement"]
    assert current == payload["tool_enforcement"]
    assert updated.json()["generation"] == 2


@pytest.mark.asyncio
async def test_simulation_streams_real_gateway_stage_outputs(api) -> None:
    client, provider = api

    response = await client.post(
        "/api/simulations",
        json={"profile": "code", "input": "Explain the flow", "max_tokens": 200},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"event": "simulation_started"' in response.text
    assert '"event": "model_completed"' in response.text
    assert '"stage": "direct"' in response.text
    assert '"content": "hello from local"' in response.text
    assert '"event": "simulation_completed"' in response.text
    assert provider.requests[-1][0] == "local-model"


@pytest.mark.asyncio
async def test_invalid_config_does_not_replace_live_generation(api) -> None:
    client, provider = api
    payload = (await client.get("/api/config")).json()["config"]
    payload["profiles"]["code"]["provider"] = "missing"

    rejected = await client.put("/api/config", json=payload)
    current = await client.get("/api/config")
    completion = await client.post(
        "/v1/chat/completions",
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "still live"}],
        },
    )

    assert rejected.status_code == 422
    assert current.json()["generation"] == 1
    assert completion.status_code == 200
    assert provider.requests[-1][0] == "local-model"


@pytest.mark.asyncio
async def test_config_swap_keeps_in_flight_request_on_its_generation(
    gateway_config,
) -> None:
    class BlockingProvider:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.models: list[str] = []

        async def complete(
            self, model: str, request: CanonicalRequest
        ) -> Completion:
            self.models.append(model)
            if model == "local-model":
                self.started.set()
                await self.release.wait()
            return Completion(content=model, model=model)

        async def stream(
            self, model: str, request: CanonicalRequest
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(content=model)
            yield StreamEvent(done=True, finish_reason="stop")

        async def close(self) -> None:
            return None

    provider = BlockingProvider()
    app = create_app(gateway_config, {"local": provider})
    transport = httpx.ASGITransport(app=app)
    request = {
        "model": "moa-code",
        "messages": [{"role": "user", "content": "run"}],
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=request))
        await provider.started.wait()
        payload = (await client.get("/api/config")).json()["config"]
        payload["profiles"]["code"]["model"] = "next-model"
        assert (await client.put("/api/config", json=payload)).status_code == 200
        second = await client.post("/v1/chat/completions", json=request)
        provider.release.set()
        first_response = await first

    assert second.json()["choices"][0]["message"]["content"] == "next-model"
    assert first_response.json()["choices"][0]["message"]["content"] == "local-model"
    assert provider.models == ["local-model", "next-model"]


@pytest.mark.asyncio
async def test_config_update_persists_when_config_path_is_set(
    gateway_config, api, tmp_path
) -> None:
    path = tmp_path / "experiment.yaml"
    _, provider = api
    app = create_app(gateway_config, {"local": provider}, config_path=path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = (await client.get("/api/config")).json()["config"]
        payload["profiles"]["code"]["model"] = "persisted-model"
        payload["tool_enforcement"] = {
            "enabled": True,
            "investigation_tools": ["task"],
            "max_investigation_calls": 2,
        }
        response = await client.put("/api/config", json=payload)

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    assert load_config(path).profiles["code"].model == "persisted-model"
    assert load_config(path).server.tool_enforcement.max_investigation_calls == 2


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
async def test_request_correlation_headers(api) -> None:
    client, _ = api

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-MoA-Parent-Request-ID": "parent:123"},
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert len(response.headers["x-moa-request-id"]) == 32
    assert response.headers["x-moa-parent-request-id"] == "parent:123"


@pytest.mark.asyncio
async def test_rejects_invalid_parent_request_id(api) -> None:
    client, provider = api

    response = await client.post(
        "/v1/chat/completions",
        headers={"X-MoA-Parent-Request-ID": "invalid value"},
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert provider.requests == []


@pytest.mark.asyncio
async def test_deepseek_style_unversioned_chat_endpoint(api) -> None:
    client, _ = api

    models = await client.get("/models")
    response = await client.post(
        "/chat/completions",
        json={
            "model": "moa-code",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert models.status_code == 200
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"


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
async def test_streaming_upstream_failure_emits_terminal_error_after_progress() -> None:
    class EmptyProvider:
        async def complete(
            self, model: str, request: CanonicalRequest
        ) -> Completion:
            return Completion(content="", model=model, finish_reason="length")

        async def stream(
            self, model: str, request: CanonicalRequest
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(finish_reason="length", done=True)

        async def close(self) -> None:
            return None

    config = GatewayConfig.model_validate(
        {
            "server": {"api_key_env": None},
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://local.test/v1",
                }
            },
            "profiles": {
                "code": {
                    "aliases": ["moa-code"],
                    "strategy": "council",
                    "contributors": [
                        {"provider": "local", "model": "one", "family": "one"},
                        {"provider": "local", "model": "two", "family": "two"},
                        {
                            "provider": "local",
                            "model": "three",
                            "family": "three",
                        },
                    ],
                    "aggregator": {"provider": "local", "model": "final"},
                    "min_quorum": 3,
                }
            },
            "default_profile": "code",
        }
    )
    app = create_app(config, {"local": EmptyProvider()})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa-code",
                "messages": [{"role": "user", "content": "solve"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert ": moa-progress collecting contributor quorum" in response.text
    assert '"code":"upstream_error"' in response.text


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
        assert (await client.get("/api/config")).status_code == 200
        assert (
            await client.post(
                "/api/simulations",
                json={"profile": "code", "input": "public simulation"},
            )
        ).status_code == 200
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
