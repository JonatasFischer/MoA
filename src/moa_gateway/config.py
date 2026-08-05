from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolEnforcementConfig(BaseModel):
    enabled: bool = False
    required_tools: list[str] = Field(default_factory=list)
    enforcement_mode: Literal["warn", "block", "auto"] = "warn"
    min_investigation_calls: int = Field(default=1, ge=1, le=8)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key_env: str | None = "MOA_API_KEY"
    trace_log_path: str | None = None
    trace_max_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    trace_backup_count: int = Field(default=3, ge=0, le=100)
    warmup_on_startup: bool = False
    warmup_profiles: list[str] = Field(default_factory=list)
    pid_file: str = ".moa.pid"
    tool_enforcement: ToolEnforcementConfig = Field(default_factory=ToolEnforcementConfig)

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


class ProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["openai-compatible", "openai", "deepseek", "ollama"] = Field(
        alias="type"
    )
    base_url: str = ""
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=1800, gt=0)

    @model_validator(mode="after")
    def validate_base_url(self) -> "ProviderConfig":
        defaults = {
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
            "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
            "ollama": ("http://127.0.0.1:11434", None),
        }
        if self.kind in defaults:
            default_url, default_key_env = defaults[self.kind]
            self.base_url = self.base_url or default_url
            self.api_key_env = self.api_key_env or default_key_env
        if not self.base_url:
            raise ValueError("openai-compatible providers require base_url")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("provider base_url must use http:// or https://")
        return self


class ModelTargetConfig(BaseModel):
    provider: str
    model: str
    role: str = "general"
    family: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    think: bool | None = None
    keep_alive: str | int | float | None = None
    num_ctx: int | None = Field(default=None, gt=0)


class ProfileConfig(BaseModel):
    aliases: list[str] = Field(min_length=1)
    strategy: Literal["direct", "classic", "council"] = "direct"
    provider: str | None = None
    model: str | None = None
    proposers: list[ModelTargetConfig] = Field(default_factory=list)
    contributors: list[ModelTargetConfig] = Field(default_factory=list)
    aggregator: ModelTargetConfig | None = None
    tool_dispatch: ModelTargetConfig | None = None
    tool_policy: Literal["final-only"] = "final-only"
    min_quorum: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=2, ge=1)
    contributor_deadline_seconds: float | None = Field(default=None, gt=0)
    proposer_max_tokens: int = Field(default=1024, ge=1)
    contributor_max_tokens: int = Field(default=1536, ge=1)
    reference_token_budget: int = Field(default=8000, ge=1)
    reasoning_reserve: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    contributor_history_chars: int = Field(default=12000, ge=1)
    contributor_format: Literal["text", "json-schema"] = "text"
    keep_alive: str | int | float | None = None
    num_ctx: int | None = Field(default=None, gt=0)
    think: bool | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> "ProfileConfig":
        if self.strategy == "direct":
            if not self.provider or not self.model:
                raise ValueError("direct profiles require provider and model")
            if self.proposers or self.contributors or self.aggregator or self.tool_dispatch:
                raise ValueError("direct profiles cannot define MoA targets")
            return self

        if self.provider or self.model:
            raise ValueError("non-direct profiles use proposers and aggregator targets")

        if self.strategy == "classic":
            if not self.proposers or not self.aggregator:
                raise ValueError("classic profiles require proposers and aggregator")
            if self.contributors:
                raise ValueError("classic profiles cannot define contributors")
            if self.min_quorum > len(self.proposers):
                raise ValueError("min_quorum cannot exceed the proposer count")
            return self

        if self.proposers:
            raise ValueError("council profiles use contributors, not proposers")
        if len(self.contributors) < 3 or not self.aggregator:
            raise ValueError(
                "council profiles require at least three contributors and an aggregator"
            )
        if self.min_quorum > len(self.contributors):
            raise ValueError("min_quorum cannot exceed the contributor count")
        families = [target.family for target in self.contributors]
        if any(not family for family in families) or len(set(families)) != len(families):
            raise ValueError(
                "council contributors require distinct non-empty model families"
            )
        return self


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
            if profile.strategy == "direct":
                provider_names = [profile.provider]
            else:
                provider_names = [
                    *(target.provider for target in profile.proposers),
                    *(target.provider for target in profile.contributors),
                    profile.aggregator.provider if profile.aggregator else None,
                    profile.tool_dispatch.provider if profile.tool_dispatch else None,
                ]
            for provider_name in (
                name for name in provider_names if name is not None
            ):
                if provider_name not in self.providers:
                    raise ValueError(
                        f"profile {name!r} references unknown provider "
                        f"{provider_name!r}"
                    )
            for alias in [name, *profile.aliases]:
                if alias in seen:
                    raise ValueError(f"duplicate profile name or alias: {alias}")
                seen.add(alias)
        unknown_warmups = set(self.server.warmup_profiles) - seen
        if unknown_warmups:
            raise ValueError(
                f"unknown warmup profile: {sorted(unknown_warmups)[0]}"
            )
        if self.server.tool_enforcement.enabled:
            if not self.server.tool_enforcement.required_tools:
                raise ValueError(
                    "tool_enforcement.enabled requires at least one required_tool"
                )
        return self

    def resolve_profile(self, model: str | None) -> tuple[str, ProfileConfig]:
        requested = model or self.default_profile
        if requested in self.profiles:
            return requested, self.profiles[requested]
        for name, profile in self.profiles.items():
            if requested in profile.aliases:
                return name, profile
        raise KeyError(requested)


class ExperimentConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    profiles: dict[str, ProfileConfig]
    default_profile: str
    tool_enforcement: ToolEnforcementConfig = Field(
        default_factory=ToolEnforcementConfig
    )

    @classmethod
    def from_gateway(cls, config: GatewayConfig) -> "ExperimentConfig":
        return cls.model_validate(
            {
                **config.model_dump(
                    include={"providers", "profiles", "default_profile"},
                    by_alias=True,
                ),
                "tool_enforcement": config.server.tool_enforcement.model_dump(),
            }
        )


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
