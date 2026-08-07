from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolEnforcementConfig(BaseModel):
    enabled: bool = False
    investigation_tools: list[str] = Field(default_factory=list)
    max_investigation_calls: int = Field(default=1, ge=1, le=8)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key_env: str | None = "MOA_API_KEY"
    trace_log_path: str | None = None
    trace_max_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    trace_backup_count: int = Field(default=3, ge=0, le=100)
    warmup_on_startup: bool = False
    warmup_profiles: list[str] = Field(default_factory=list)
    warmup_flows: list[str] = Field(default_factory=list)
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
    input_modalities: list[Literal["text", "image", "file"]] = Field(
        default_factory=lambda: ["text"]
    )
    model_input_modalities: dict[
        str, list[Literal["text", "image", "file"]]
    ] = Field(default_factory=dict)

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
        if "text" not in self.input_modalities:
            raise ValueError("provider input_modalities must include text")
        if len(set(self.input_modalities)) != len(self.input_modalities):
            raise ValueError("provider input_modalities must be unique")
        for model, modalities in self.model_input_modalities.items():
            if not model or "text" not in modalities:
                raise ValueError("model_input_modalities must include text")
            if len(set(modalities)) != len(modalities):
                raise ValueError("model_input_modalities must be unique")
        return self

    def modalities_for(self, model: str) -> set[str]:
        return set(self.model_input_modalities.get(model, self.input_modalities))


# Legacy profile types remain available while programmatic callers migrate to flows.
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


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str
    context: str | None = None


class ToolTransformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force_arguments: dict[str, Any] = Field(default_factory=dict)
    prepend_latest_user_request: bool = False
    prompt_field: str = "prompt"
    required_prompt_marker: str | None = None
    append_prompt_if_missing: str | None = None


class ToolValidatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_tools: Literal["client"] | list[str] = "client"
    require_client_definition: bool = True
    require_call_id: bool = True
    arguments: Literal["json-object"] = "json-object"
    mixed_text: Literal["preserve", "discard"] = "preserve"
    execute: bool = False
    transforms: dict[str, ToolTransformConfig] = Field(default_factory=dict)


class StepTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    when: Literal["always", "has_tool_calls", "no_tool_calls"] = "always"


class StepToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "client"] = "none"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    validator: str | None = None
    max_calls: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_filters(self) -> "StepToolsConfig":
        if self.include and self.exclude:
            raise ValueError("step tools cannot define both include and exclude")
        if self.max_calls is not None and not self.include:
            raise ValueError("max_calls requires an explicit tool include list")
        return self


class RepairConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    attempts: int = Field(default=1, ge=1, le=4)


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = Field(default=1, ge=1, le=4)
    condition: Literal["empty"] = "empty"
    max_tokens_multiplier: float = Field(default=2, gt=0)
    think: bool | None = None


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate: str
    strategy: Literal["best-nonempty"] = "best-nonempty"


class AiStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["ai"]
    prompt: str | None = None
    prompt_variables: dict[str, str] = Field(default_factory=dict)
    provider: str
    model: str
    role: str = "general"
    family: str | None = None
    conversation: Literal["none", "advisory", "full"] = "none"
    activation: Literal["single", "first"] = "single"
    max_tokens: int | Literal["request"] | None = None
    reasoning_reserve: int = Field(default=0, ge=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    think: bool | None = None
    keep_alive: str | int | float | None = None
    num_ctx: int | None = Field(default=None, gt=0)
    response_schema: str | None = None
    repair: RepairConfig | None = None
    tools: StepToolsConfig = Field(default_factory=StepToolsConfig)
    retry: RetryConfig | None = None
    fallback: FallbackConfig | None = None
    targets: list[StepTargetConfig] = Field(default_factory=list)


class GateStepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["gate"]
    min_success: int = Field(ge=1)
    max_concurrency: int = Field(default=1, ge=1)
    deadline_seconds: float | None = Field(default=None, gt=0)
    completion: Literal["all-or-deadline"] = "all-or-deadline"
    on_failure: Literal["fail"] = "fail"
    targets: list[StepTargetConfig] = Field(default_factory=list)


FlowStepConfig = Annotated[AiStepConfig | GateStepConfig, Field(discriminator="type")]


class FlowStartConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    priority: int | None = Field(default=None, ge=1)
    when: Literal[
        "always",
        "skill_result",
        "investigation_result",
        "tool_continuation",
        "opencode_maintenance",
        "delegated_investigation",
        "simple_request",
    ] = "always"


class FlowOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    passthrough_input_on_no_tool_calls: bool = False


class FlowRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_latest_user_chars: int = Field(default=800, ge=1)
    max_conversation_chars: int = Field(default=4000, ge=1)
    max_messages: int = Field(default=4, ge=1)
    require_no_tools: bool = True


class FlowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: list[str] = Field(min_length=1)
    starts: list[FlowStartConfig] = Field(min_length=1)
    output: FlowOutputConfig
    routing: FlowRoutingConfig | None = None
    steps: list[FlowStepConfig] = Field(min_length=1)

    @property
    def step_map(self) -> dict[str, FlowStepConfig]:
        return {step.id: step for step in self.steps}


class GatewayConfig(BaseModel):
    version: Literal[1, 2] = 1
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: dict[str, ProviderConfig]
    prompts: dict[str, PromptConfig] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_validators: dict[str, ToolValidatorConfig] = Field(default_factory=dict)
    flows: dict[str, FlowConfig] = Field(default_factory=dict)
    default_flow: str | None = None
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    default_profile: str | None = None

    @property
    def uses_flows(self) -> bool:
        return bool(self.flows)

    @model_validator(mode="after")
    def validate_references(self) -> "GatewayConfig":
        if self.flows:
            self._validate_flows()
        elif self.profiles:
            self._validate_profiles()
        else:
            raise ValueError("configuration requires flows or profiles")
        return self

    def _validate_profiles(self) -> None:
        if not self.default_profile or self.default_profile not in self.profiles:
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
            for provider_name in (item for item in provider_names if item is not None):
                if provider_name not in self.providers:
                    raise ValueError(
                        f"profile {name!r} references unknown provider {provider_name!r}"
                    )
            self._validate_aliases(name, profile.aliases, seen)
        unknown = set(self.server.warmup_profiles) - seen
        if unknown:
            raise ValueError(f"unknown warmup profile: {sorted(unknown)[0]}")
        if self.server.tool_enforcement.enabled:
            if not self.server.tool_enforcement.investigation_tools:
                raise ValueError(
                    "tool_enforcement.enabled requires at least one investigation_tool"
                )

    def _validate_flows(self) -> None:
        if self.version != 2:
            raise ValueError("flow configuration requires version: 2")
        if not self.default_flow or self.default_flow not in self.flows:
            raise ValueError(f"unknown default_flow: {self.default_flow}")
        for schema_name, schema in self.schemas.items():
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise ValueError(f"invalid schema {schema_name!r}: {exc.message}") from exc
        seen: set[str] = set()
        for name, flow in self.flows.items():
            self._validate_aliases(name, flow.aliases, seen)
            self._validate_flow(name, flow)
        unknown = set(self.server.warmup_flows) - seen
        if unknown:
            raise ValueError(f"unknown warmup flow: {sorted(unknown)[0]}")

    @staticmethod
    def _validate_aliases(name: str, aliases: list[str], seen: set[str]) -> None:
        for alias in [name, *aliases]:
            if alias in seen:
                raise ValueError(f"duplicate flow/profile name or alias: {alias}")
            seen.add(alias)

    def _validate_flow(self, name: str, flow: FlowConfig) -> None:
        step_map = flow.step_map
        if len(step_map) != len(flow.steps):
            raise ValueError(f"flow {name!r} has duplicate step ids")
        if flow.output.step not in step_map:
            raise ValueError(f"flow {name!r} has unknown output step {flow.output.step!r}")
        for start in flow.starts:
            if start.step not in step_map:
                raise ValueError(f"flow {name!r} has unknown start step {start.step!r}")
        if not any(start.when == "always" for start in flow.starts):
            raise ValueError(f"flow {name!r} requires an always start fallback")
        if any(start.when == "simple_request" for start in flow.starts) and not flow.routing:
            raise ValueError(
                f"flow {name!r} simple_request start requires routing configuration"
            )

        predecessors: dict[str, set[str]] = {step_id: set() for step_id in step_map}
        adjacency: dict[str, set[str]] = {step_id: set() for step_id in step_map}
        source_gate: dict[str, str] = {}
        for step in flow.steps:
            if isinstance(step, AiStepConfig):
                if step.provider not in self.providers:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} references unknown provider "
                        f"{step.provider!r}"
                    )
                if step.prompt and step.prompt not in self.prompts:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} references unknown prompt "
                        f"{step.prompt!r}"
                    )
                if step.response_schema and step.response_schema not in self.schemas:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} references unknown schema "
                        f"{step.response_schema!r}"
                    )
                if step.repair and step.repair.prompt not in self.prompts:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} references unknown repair prompt"
                    )
                if step.tools.validator and step.tools.validator not in self.tool_validators:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} references unknown tool validator"
                    )
                if step.tools.validator:
                    validator = self.tool_validators[step.tools.validator]
                    if validator.transforms and step.tools.max_calls is None:
                        raise ValueError(
                            f"flow {name!r} step {step.id!r} must configure max_calls "
                            "for transformed investigation tools"
                        )
                if step.fallback and step.fallback.gate not in step_map:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} has unknown fallback gate"
                    )
            target_keys: set[tuple[str, str]] = set()
            target_steps: set[str] = set()
            for target in step.targets:
                target_key = (target.step, target.when)
                if target_key in target_keys:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} has a duplicate target"
                    )
                target_keys.add(target_key)
                if target.step in target_steps:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} targets the same step more "
                        "than once"
                    )
                target_steps.add(target.step)
                if target.step == "$return":
                    continue
                if target.step not in step_map:
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} targets unknown step "
                        f"{target.step!r}"
                    )
                adjacency[step.id].add(target.step)
                predecessors[target.step].add(step.id)
                if isinstance(step_map[target.step], GateStepConfig):
                    if target.when != "always":
                        raise ValueError(
                            f"flow {name!r} gate {target.step!r} cannot have "
                            "conditional sources"
                        )
                    previous_gate = source_gate.get(step.id)
                    if previous_gate and previous_gate != target.step:
                        raise ValueError(
                            f"flow {name!r} step {step.id!r} cannot feed multiple gates"
                        )
                    source_gate[step.id] = target.step

            conditions = {target.when for target in step.targets}
            if (
                step.targets
                and "always" not in conditions
                and not {"has_tool_calls", "no_tool_calls"}.issubset(conditions)
            ):
                raise ValueError(
                    f"flow {name!r} step {step.id!r} has incomplete conditional targets"
                )

        for step in flow.steps:
            if isinstance(step, GateStepConfig):
                source_count = len(predecessors[step.id])
                if source_count == 0:
                    raise ValueError(f"flow {name!r} gate {step.id!r} has no sources")
                if step.min_success > source_count:
                    raise ValueError(
                        f"flow {name!r} gate {step.id!r} min_success exceeds sources"
                    )
            elif len(predecessors[step.id]) > 1 and step.activation != "first":
                raise ValueError(
                    f"flow {name!r} AI step {step.id!r} has multiple sources and "
                    "must configure activation: first or use a gate"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError(f"flow {name!r} contains a cycle at {step_id!r}")
            if step_id in visited:
                return
            visiting.add(step_id)
            for target_id in adjacency[step_id]:
                visit(target_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for start in flow.starts:
            visit(start.step)
        unreachable = set(step_map) - visited
        if unreachable:
            raise ValueError(
                f"flow {name!r} has unreachable step {sorted(unreachable)[0]!r}"
            )

        return_sources = {
            step.id
            for step in flow.steps
            if any(target.step == "$return" for target in step.targets)
        }
        can_return = set(return_sources)
        changed = True
        while changed:
            changed = False
            for source_id, targets in adjacency.items():
                if source_id not in can_return and targets & can_return:
                    can_return.add(source_id)
                    changed = True
        stranded = visited - can_return
        if stranded:
            raise ValueError(
                f"flow {name!r} step {sorted(stranded)[0]!r} cannot reach $return"
            )

        def reaches(source_id: str, target_id: str) -> bool:
            pending = [source_id]
            seen_reach: set[str] = set()
            while pending:
                current = pending.pop()
                if current == target_id:
                    return True
                if current in seen_reach:
                    continue
                seen_reach.add(current)
                pending.extend(adjacency[current])
            return False

        for step in flow.steps:
            if (
                isinstance(step, AiStepConfig)
                and step.fallback
            ):
                fallback_source = step_map[step.fallback.gate]
                if not isinstance(fallback_source, GateStepConfig):
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} fallback must reference a gate"
                    )
                if not reaches(step.fallback.gate, step.id):
                    raise ValueError(
                        f"flow {name!r} step {step.id!r} fallback gate must be an ancestor"
                    )

    def resolve_profile(self, model: str | None) -> tuple[str, ProfileConfig]:
        if not self.profiles:
            raise KeyError(model or self.default_flow or "")
        requested = model or self.default_profile
        if requested in self.profiles:
            return str(requested), self.profiles[str(requested)]
        for name, profile in self.profiles.items():
            if requested in profile.aliases:
                return name, profile
        raise KeyError(requested)

    def resolve_flow(self, model: str | None) -> tuple[str, FlowConfig]:
        requested = model or self.default_flow
        if requested in self.flows:
            return str(requested), self.flows[str(requested)]
        for name, flow in self.flows.items():
            if requested in flow.aliases:
                return name, flow
        raise KeyError(requested)


class ExperimentConfig(BaseModel):
    version: Literal[1, 2] = 1
    warmup_flows: list[str] = Field(default_factory=list)
    warmup_profiles: list[str] = Field(default_factory=list)
    providers: dict[str, ProviderConfig]
    prompts: dict[str, PromptConfig] = Field(default_factory=dict)
    schemas: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_validators: dict[str, ToolValidatorConfig] = Field(default_factory=dict)
    flows: dict[str, FlowConfig] = Field(default_factory=dict)
    default_flow: str | None = None
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    default_profile: str | None = None
    tool_enforcement: ToolEnforcementConfig = Field(
        default_factory=ToolEnforcementConfig
    )

    @classmethod
    def from_gateway(cls, config: GatewayConfig) -> "ExperimentConfig":
        include = {
            "version",
            "providers",
            "prompts",
            "schemas",
            "tool_validators",
            "flows",
            "default_flow",
            "profiles",
            "default_profile",
        }
        return cls.model_validate(
            {
                **config.model_dump(include=include, by_alias=True),
                "warmup_flows": config.server.warmup_flows,
                "warmup_profiles": config.server.warmup_profiles,
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
