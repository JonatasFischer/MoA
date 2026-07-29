from __future__ import annotations

from collections.abc import AsyncIterator

from moa_gateway.config import GatewayConfig
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent
from moa_gateway.provider import OpenAICompatibleProvider, Provider


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider] | None = None,
    ) -> None:
        self.config = config
        self.providers: dict[str, Provider] = providers or {
            name: OpenAICompatibleProvider(provider)
            for name, provider in config.providers.items()
        }

    def public_model(self, request: CanonicalRequest) -> str:
        _, profile = self.config.resolve_profile(request.requested_model)
        return request.requested_model or profile.aliases[0]

    async def complete(self, request: CanonicalRequest) -> Completion:
        _, profile = self.config.resolve_profile(request.requested_model)
        return await self.providers[profile.provider].complete(profile.model, request)

    async def stream(self, request: CanonicalRequest) -> AsyncIterator[StreamEvent]:
        _, profile = self.config.resolve_profile(request.requested_model)
        async for event in self.providers[profile.provider].stream(
            profile.model, request
        ):
            yield event

    async def close(self) -> None:
        for provider in self.providers.values():
            await provider.close()
