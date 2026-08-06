from __future__ import annotations

import pytest
from pydantic import ValidationError

from moa_gateway.config import GatewayConfig, ProviderConfig, load_config


def test_loads_yaml_and_resolves_alias(tmp_path) -> None:
    path = tmp_path / "moa.yaml"
    path.write_text(
        """
providers:
  local:
    type: openai-compatible
    base_url: http://127.0.0.1:11434/v1
profiles:
  code:
    aliases: [claude-moa-code]
    strategy: direct
    provider: local
    model: local-model
default_profile: code
""",
        encoding="utf-8",
    )

    config = load_config(path)

    name, profile = config.resolve_profile("claude-moa-code")
    assert name == "code"
    assert profile.model == "local-model"


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError, match="unknown provider"):
        GatewayConfig.model_validate(
            {
                "providers": {
                    "local": {
                        "type": "openai-compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                    }
                },
                "profiles": {
                    "code": {
                        "aliases": ["moa-code"],
                        "provider": "missing",
                        "model": "model",
                    }
                },
                "default_profile": "code",
            }
        )


def test_rejects_duplicate_alias() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        GatewayConfig.model_validate(
            {
                "providers": {
                    "local": {
                        "type": "openai-compatible",
                        "base_url": "http://127.0.0.1:11434/v1",
                    }
                },
                "profiles": {
                    "one": {
                        "aliases": ["same"],
                        "provider": "local",
                        "model": "model",
                    },
                    "two": {
                        "aliases": ["same"],
                        "provider": "local",
                        "model": "model",
                    },
                },
                "default_profile": "one",
            }
        )


def test_validates_classic_profile_targets_and_quorum() -> None:
    config = GatewayConfig.model_validate(
        {
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://127.0.0.1:11434/v1",
                }
            },
            "profiles": {
                "code": {
                    "aliases": ["moa-code"],
                    "strategy": "classic",
                    "proposers": [
                        {"provider": "local", "model": "small", "role": "reviewer"}
                    ],
                    "aggregator": {"provider": "local", "model": "large"},
                    "min_quorum": 1,
                }
            },
            "default_profile": "code",
        }
    )

    assert config.profiles["code"].aggregator.model == "large"

    broken = config.model_dump(by_alias=True)
    broken["profiles"]["code"]["min_quorum"] = 2
    with pytest.raises(ValidationError, match="min_quorum"):
        GatewayConfig.model_validate(broken)


@pytest.mark.parametrize(
    ("kind", "base_url", "api_key_env"),
    [
        ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        ("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    ],
)
def test_named_provider_defaults(kind, base_url, api_key_env) -> None:
    provider = ProviderConfig.model_validate({"type": kind})

    assert provider.base_url == base_url
    assert provider.api_key_env == api_key_env


def test_native_ollama_provider_defaults() -> None:
    provider = ProviderConfig.model_validate({"type": "ollama"})

    assert provider.base_url == "http://127.0.0.1:11434"
    assert provider.api_key_env is None


def test_deepseek_can_be_the_classic_aggregator() -> None:
    config = GatewayConfig.model_validate(
        {
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://127.0.0.1:11434/v1",
                },
                "judge": {"type": "deepseek"},
            },
            "profiles": {
                "code": {
                    "aliases": ["deepseek-moa-code"],
                    "strategy": "classic",
                    "proposers": [{"provider": "local", "model": "local-model"}],
                    "aggregator": {
                        "provider": "judge",
                        "model": "deepseek-chat",
                        "role": "final judge",
                    },
                }
            },
            "default_profile": "code",
        }
    )

    assert config.profiles["code"].aggregator.provider == "judge"


def test_council_requires_three_complete_contributors() -> None:
    profile = {
        "aliases": ["moa-code"],
        "strategy": "council",
        "contributors": [
            {"provider": "local", "model": "qwen-small", "family": "qwen"},
            {"provider": "local", "model": "gemma-small", "family": "gemma"},
            {
                "provider": "local",
                "model": "deepseek-small",
                "family": "deepseek",
            },
        ],
        "aggregator": {"provider": "local", "model": "qwen3.6:27b"},
        "min_quorum": 3,
    }
    raw = {
        "providers": {
            "local": {
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
            }
        },
        "profiles": {"code": profile},
        "default_profile": "code",
    }

    config = GatewayConfig.model_validate(raw)
    assert config.profiles["code"].aggregator.model == "qwen3.6:27b"

    profile["contributors"].pop()
    profile["min_quorum"] = 2
    with pytest.raises(ValidationError, match="at least three contributors"):
        GatewayConfig.model_validate(raw)


def test_v2_code_flow_preserves_council_targets_and_gate_contract() -> None:
    config = load_config("moa.yaml")
    flow = config.flows["code"]
    steps = flow.step_map

    assert {steps[name].family for name in (
        "qwen-council",
        "gemma-council",
        "deepseek-council",
    )} == {
        "Opus",
        "Mitos",
        "fable",
    }
    assert steps["aggregate"].model == "Qwen/Qwen3-Coder-Next-FP8"
    assert steps["aggregate"].think is True
    assert steps["integrate-investigation"].model == "Qwen/Qwen3-Coder-Next-FP8"
    assert steps["aggregate"].reasoning_reserve == 4096
    assert steps["contributions"].min_success == 2
    assert steps["contributions"].deadline_seconds == 45
    assert steps["contributions"].max_concurrency == 3
    assert steps["investigation-check"].tools.max_calls == 3

    broken = config.model_dump(by_alias=True)
    gate = next(
        step for step in broken["flows"]["code"]["steps"]
        if step["id"] == "contributions"
    )
    gate["min_success"] = 4
    with pytest.raises(ValidationError, match="min_success exceeds sources"):
        GatewayConfig.model_validate(broken)

    broken = config.model_dump(by_alias=True)
    broken["server"]["warmup_flows"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown warmup flow"):
        GatewayConfig.model_validate(broken)

    broken = config.model_dump(by_alias=True)
    checker = next(
        step for step in broken["flows"]["code"]["steps"]
        if step["id"] == "investigation-check"
    )
    checker["tools"]["max_calls"] = None
    with pytest.raises(ValidationError, match="must configure max_calls"):
        GatewayConfig.model_validate(broken)


def test_provider_modalities_require_text_and_unique_values() -> None:
    with pytest.raises(ValueError, match="must include text"):
        ProviderConfig.model_validate(
            {
                "type": "openai-compatible",
                "base_url": "http://local.test/v1",
                "input_modalities": ["image"],
            }
        )

    with pytest.raises(ValueError, match="must be unique"):
        ProviderConfig.model_validate(
            {
                "type": "openai-compatible",
                "base_url": "http://local.test/v1",
                "input_modalities": ["text", "text"],
            }
        )


def test_simple_request_start_requires_routing_configuration() -> None:
    raw = load_config("moa.yaml").model_dump(by_alias=True)
    raw["flows"]["direct"]["starts"].insert(
        0, {"step": "answer", "when": "simple_request"}
    )

    with pytest.raises(ValueError, match="requires routing configuration"):
        GatewayConfig.model_validate(raw)
