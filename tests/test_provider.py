from __future__ import annotations

import json

import httpx
import pytest

from moa_gateway.config import ProviderConfig
from moa_gateway.domain import CanonicalRequest, ProviderMetrics
from moa_gateway.provider import (
    DeepSeekProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OllamaProvider,
    UpstreamError,
    create_provider,
    discover_models,
)


TOOL_CALL = {
    "id": "call_123",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
}


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
        think=False,
        response_format={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
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
    assert requests[0]["think"] is False
    assert requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "strict": True,
            "schema": canonical.response_format,
        },
    }
    assert requests[1]["think"] is False
    assert requests[1]["response_format"] == requests[0]["response_format"]
    assert requests[1]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_openai_compatible_tool_completion() -> None:
    request_payload: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        request_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "local-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [TOOL_CALL],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
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
        messages=[{"role": "user", "content": "read"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
    )

    completion = await provider.complete("local-model", canonical)
    await provider.close()

    assert completion.tool_calls == [TOOL_CALL]
    assert completion.finish_reason == "tool_calls"
    assert request_payload["tools"] == canonical.tools
    assert request_payload["tool_choice"] == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "provider_class", "api_key_env"),
    [
        ("openai", OpenAIProvider, "OPENAI_API_KEY"),
        ("deepseek", DeepSeekProvider, "DEEPSEEK_API_KEY"),
    ],
)
async def test_named_provider_adapters(
    monkeypatch, kind, provider_class, api_key_env
) -> None:
    monkeypatch.setenv(api_key_env, "secret")

    provider = create_provider(ProviderConfig.model_validate({"type": kind}))

    assert isinstance(provider, provider_class)
    assert provider.client.headers["authorization"] == "Bearer secret"
    await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "response_body", "expected_path", "expected_models"),
    [
        (
            "ollama",
            {"models": [{"name": "qwen"}, {"name": "gemma"}]},
            "/api/tags",
            ["gemma", "qwen"],
        ),
        (
            "openai-compatible",
            {"data": [{"id": "gpt-b"}, {"id": "gpt-a"}]},
            "/v1/models",
            ["gpt-a", "gpt-b"],
        ),
    ],
)
async def test_discover_models(
    monkeypatch, kind, response_body, expected_path, expected_models
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body)

    async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return async_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("moa_gateway.provider.httpx.AsyncClient", client_factory)
    config = ProviderConfig.model_validate(
        {
            "type": kind,
            "base_url": (
                "http://provider.test"
                if kind == "ollama"
                else "http://provider.test/v1"
            ),
        }
    )

    models = await discover_models(config)

    assert models == expected_models
    assert requests[0].url.path == expected_path


@pytest.mark.asyncio
async def test_native_ollama_complete_and_ndjson_stream() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body["stream"]:
            content = (
                '{"message":{"role":"assistant","content":"hel"},"done":false}\n'
                '{"message":{"role":"assistant","content":"lo"},"done":false}\n'
                '{"message":{"role":"assistant","content":""},"done":true,'
                '"done_reason":"stop","prompt_eval_count":7,"eval_count":2,'
                '"total_duration":100,"load_duration":20,'
                '"prompt_eval_duration":30,"eval_duration":50}\n'
            )
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(
            200,
            json={
                "model": "qwen",
                "message": {"role": "assistant", "content": "hello"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 2,
                "total_duration": 100,
                "load_duration": 20,
                "prompt_eval_duration": 30,
                "eval_duration": 50,
            },
        )

    provider = OllamaProvider(
        ProviderConfig.model_validate(
            {"type": "ollama", "base_url": "http://ollama.test"}
        )
    )
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="http://ollama.test/", transport=httpx.MockTransport(handler)
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    canonical = CanonicalRequest(
        requested_model="moa-code",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=12,
        temperature=0.2,
        stop="END",
        think=False,
        keep_alive=-1,
        num_ctx=32768,
        response_format=schema,
    )

    completion = await provider.complete("qwen", canonical)
    events = [event async for event in provider.stream("qwen", canonical)]
    await provider.close()

    assert completion.content == "hello"
    assert completion.usage.total_tokens == 9
    assert completion.metrics == ProviderMetrics(100, 20, 30, 50)
    assert [event.content for event in events if event.content] == ["hel", "lo"]
    assert events[-1].done is True
    assert events[-1].metrics == ProviderMetrics(100, 20, 30, 50)
    assert requests[0]["options"] == {
        "num_predict": 12,
        "temperature": 0.2,
        "stop": ["END"],
        "num_ctx": 32768,
    }
    assert requests[0]["think"] is False
    assert requests[0]["keep_alive"] == -1
    assert requests[0]["format"] == schema


def test_native_ollama_normalizes_tool_arguments() -> None:
    request = CanonicalRequest(
        requested_model=None,
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "text"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
    )

    payload = OllamaProvider._payload("qwen", request, stream=False)

    assert payload["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "README.md"
    }
    assert payload["messages"][1]["tool_name"] == "read"
    assert payload["tools"] == request.tools


@pytest.mark.asyncio
async def test_native_ollama_rejects_stream_without_done() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content='{"message":{"content":"partial"},"done":false}\n',
        )

    provider = OllamaProvider(ProviderConfig.model_validate({"type": "ollama"}))
    await provider.client.aclose()
    provider.client = httpx.AsyncClient(
        base_url="http://ollama.test/", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(UpstreamError, match="before done=true"):
        _ = [
            event
            async for event in provider.stream(
                "qwen", CanonicalRequest(None, [{"role": "user", "content": "hi"}])
            )
        ]
    await provider.close()
