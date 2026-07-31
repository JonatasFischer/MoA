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


def test_council_requires_all_distinct_contributor_families() -> None:
    config = load_config("moa.yaml")
    profile = config.profiles["code"]

    assert {target.family for target in profile.contributors} == {
        "qwen",
        "gemma",
        "deepseek",
    }
    assert profile.aggregator.model == "qwen3.6:27b"
    assert profile.aggregator.think is False
    assert profile.tool_dispatch.model == "qwen3.6:27b"
    assert profile.reasoning_reserve == {"qwen": 4096}

    broken = config.model_dump(by_alias=True)
    broken["profiles"]["code"]["contributors"][2]["family"] = "qwen"
    with pytest.raises(ValidationError, match="distinct non-empty"):
        GatewayConfig.model_validate(broken)

    assert profile.min_quorum == 2
    assert profile.contributor_deadline_seconds == 45
    assert profile.max_concurrency == 1
    assert profile.contributor_format == "json-schema"

    broken = config.model_dump(by_alias=True)
    broken["profiles"]["code"]["min_quorum"] = 4
    with pytest.raises(ValidationError, match="contributor count"):
        GatewayConfig.model_validate(broken)

    broken = config.model_dump(by_alias=True)
    broken["server"]["warmup_profiles"] = ["missing"]
    with pytest.raises(ValidationError, match="unknown warmup profile"):
        GatewayConfig.model_validate(broken)
