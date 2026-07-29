from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from moa_gateway.app import create_app
from moa_gateway.config import GatewayConfig
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent, Usage


class FakeProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[str, CanonicalRequest]] = []
        self.closed = False

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        self.requests.append((model, request))
        return Completion(
            content="hello from local",
            model=model,
            usage=Usage(input_tokens=7, output_tokens=3),
        )

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append((model, request))
        yield StreamEvent(content="hello ")
        yield StreamEvent(content="from local")
        yield StreamEvent(
            finish_reason="stop",
            usage=Usage(input_tokens=7, output_tokens=3),
            done=True,
        )

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {"api_key_env": None},
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://ollama.test/v1",
                }
            },
            "profiles": {
                "code": {
                    "aliases": ["claude-moa-code", "moa-code"],
                    "strategy": "direct",
                    "provider": "local",
                    "model": "local-model",
                }
            },
            "default_profile": "code",
        }
    )


@pytest.fixture
async def api(gateway_config: GatewayConfig):
    provider = FakeProvider()
    app = create_app(gateway_config, {"local": provider})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, provider
