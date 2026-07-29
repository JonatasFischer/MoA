from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key_env: str | None = "MOA_API_KEY"

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["openai-compatible"] = Field(alias="type")
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=300, gt=0)

    @model_validator(mode="after")
    def validate_base_url(self) -> "ProviderConfig":
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("provider base_url must use http:// or https://")
        return self


class ProfileConfig(BaseModel):
    aliases: list[str] = Field(min_length=1)
    strategy: Literal["direct"] = "direct"
    provider: str
    model: str


class GatewayConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: dict[str, ProviderConfig]
    profiles: dict[str, ProfileConfig]
    default_profile: str

    @model_validator(mode="after")
    def validate_references(self) -> "GatewayConfig":
        if self.default_profile not in self.profiles:
            raise ValueError(f"unknown default_profile: {self.default_profile}")

        seen: set[str] = set()
        for name, profile in self.profiles.items():
            if profile.provider not in self.providers:
                raise ValueError(
                    f"profile {name!r} references unknown provider {profile.provider!r}"
                )
            for alias in [name, *profile.aliases]:
                if alias in seen:
                    raise ValueError(f"duplicate profile name or alias: {alias}")
                seen.add(alias)
        return self

    def resolve_profile(self, model: str | None) -> tuple[str, ProfileConfig]:
        requested = model or self.default_profile
        if requested in self.profiles:
            return requested, self.profiles[requested]
        for name, profile in self.profiles.items():
            if requested in profile.aliases:
                return name, profile
        raise KeyError(requested)


def load_config(path: str | Path | None = None) -> GatewayConfig:
    config_path = Path(path or os.getenv("MOA_CONFIG", "moa.yaml"))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"configuration root in {config_path} must be an object")
    return GatewayConfig.model_validate(raw)
