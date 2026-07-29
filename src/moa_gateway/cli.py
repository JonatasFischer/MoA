from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
import yaml
from pydantic import ValidationError

from moa_gateway.app import create_app
from moa_gateway.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moa", description="Local MoA gateway")
    parser.add_argument("--config", default="moa.yaml", help="configuration file")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a starter configuration")
    init.add_argument(
        "--model", default="qwen2.5-coder:7b", help="Ollama model id"
    )

    config = commands.add_parser("config", help="configuration operations")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("validate", help="validate configuration")

    commands.add_parser("serve", help="run the gateway")
    return parser


def _starter_config(model: str) -> dict[str, object]:
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "api_key_env": "MOA_API_KEY",
        },
        "providers": {
            "ollama": {
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "timeout_seconds": 300,
            }
        },
        "profiles": {
            "code": {
                "aliases": ["claude-moa-code", "moa-code"],
                "strategy": "direct",
                "provider": "ollama",
                "model": model,
            }
        },
        "default_profile": "code",
    }


def main() -> None:
    args = _parser().parse_args()
    path = Path(args.config)

    if args.command == "init":
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing configuration: {path}")
        path.write_text(
            yaml.safe_dump(_starter_config(args.model), sort_keys=False),
            encoding="utf-8",
        )
        print(f"created {path}")
        return

    try:
        config = load_config(path)
    except (ValueError, ValidationError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.command == "config" and args.config_command == "validate":
        print(
            f"valid: {len(config.providers)} provider(s), "
            f"{len(config.profiles)} profile(s)"
        )
        return

    if args.command == "serve":
        uvicorn.run(
            create_app(config),
            host=config.server.host,
            port=config.server.port,
        )


if __name__ == "__main__":
    main()
