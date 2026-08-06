from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import yaml

from moa_gateway.config import ExperimentConfig, GatewayConfig
from moa_gateway.gateway import Gateway
from moa_gateway.provider import Provider


class GatewayRuntime:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self._gateway = Gateway(config, providers)
        self._provider_overrides = providers
        self._config_path = Path(config_path) if config_path is not None else None
        self._generation = 1
        self._lock = asyncio.Lock()
        self._active: dict[Gateway, int] = {}
        self._retired: set[Gateway] = set()

    @property
    def config(self) -> GatewayConfig:
        return self._gateway.config

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def persisted(self) -> bool:
        return self._config_path is not None

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[Gateway]:
        async with self._lock:
            gateway = self._gateway
            self._active[gateway] = self._active.get(gateway, 0) + 1
        try:
            yield gateway
        finally:
            close_gateway = False
            async with self._lock:
                remaining = self._active[gateway] - 1
                if remaining:
                    self._active[gateway] = remaining
                else:
                    del self._active[gateway]
                    if gateway in self._retired:
                        self._retired.remove(gateway)
                        close_gateway = True
            if close_gateway:
                await gateway.close()

    async def reconfigure(self, experiment: ExperimentConfig) -> GatewayConfig:
        close_gateway: Gateway | None = None
        async with self._lock:
            current = self._gateway.config
            server = current.server.model_copy(deep=True)
            server.tool_enforcement = experiment.tool_enforcement.model_copy(deep=True)
            server.warmup_flows = list(experiment.warmup_flows)
            server.warmup_profiles = list(experiment.warmup_profiles)
            updated = GatewayConfig.model_validate(
                {
                    "version": experiment.version,
                    "server": server.model_dump(),
                    **experiment.model_dump(by_alias=True),
                }
            )
            replacement = Gateway(updated, self._provider_overrides)
            try:
                if self._config_path is not None:
                    self._write_config(updated)
            except Exception:
                if self._provider_overrides is None:
                    await replacement.close()
                raise

            previous = self._gateway
            self._gateway = replacement
            self._generation += 1
            if self._provider_overrides is None:
                if self._active.get(previous):
                    self._retired.add(previous)
                else:
                    close_gateway = previous

        if close_gateway is not None:
            await close_gateway.close()
        return updated

    def _write_config(self, config: GatewayConfig) -> None:
        assert self._config_path is not None
        temporary = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
        payload = yaml.safe_dump(
            config.model_dump(by_alias=True, exclude_none=True),
            sort_keys=False,
        )
        temporary.write_text(payload, encoding="utf-8")
        if self._config_path.exists():
            mode = stat.S_IMODE(self._config_path.stat().st_mode)
            os.chmod(temporary, mode)
        temporary.replace(self._config_path)

    async def warmup(self) -> None:
        async with self.lease() as gateway:
            await gateway.warmup()

    async def close(self) -> None:
        async with self._lock:
            gateways = {self._gateway, *self._retired}
            self._retired.clear()
        if self._provider_overrides is not None:
            await self._gateway.close()
            return
        await asyncio.gather(*(gateway.close() for gateway in gateways))
