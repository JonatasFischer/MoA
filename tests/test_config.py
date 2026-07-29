from __future__ import annotations

import pytest
from pydantic import ValidationError

from moa_gateway.config import GatewayConfig, load_config


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
