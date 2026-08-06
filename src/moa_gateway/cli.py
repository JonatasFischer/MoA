from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

import httpx
import uvicorn
import yaml
from pydantic import ValidationError

from moa_gateway.app import create_app
from moa_gateway.benchmark import run_benchmark
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
    commands.add_parser("status", help="show gateway process and health status")

    benchmark = commands.add_parser(
        "benchmark", help="benchmark direct and MoA coding quality"
    )
    benchmark.add_argument(
        "--profiles",
        nargs="+",
        default=["direct", "council-k2", "council-k3", "self-consistency"],
        help="profiles to test",
    )
    benchmark.add_argument("--runs", type=int, default=1, help="runs per case")
    benchmark.add_argument(
        "--output", type=Path, default=Path("benchmark-results.json")
    )
    benchmark.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="confirm execution of AST-screened generated code",
    )
    return parser


def _starter_config(model: str) -> dict[str, object]:
    return {
        "version": 2,
        "server": {
            "host": "127.0.0.1",
            "port": 8080,
            "api_key_env": "MOA_API_KEY",
            "trace_log_path": "moa-trace.jsonl",
        },
        "providers": {
            "ollama": {
                "type": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "timeout_seconds": 1800,
            }
        },
        "tool_validators": {
            "client-tools": {
                "allowed_tools": "client",
                "require_client_definition": True,
                "require_call_id": True,
                "arguments": "json-object",
                "mixed_text": "preserve",
            }
        },
        "flows": {
            "code": {
                "aliases": ["claude-moa-code", "moa-code"],
                "starts": [{"step": "answer", "when": "always"}],
                "output": {"step": "answer"},
                "steps": [
                    {
                        "id": "answer",
                        "type": "ai",
                        "provider": "ollama",
                        "model": model,
                        "conversation": "full",
                        "tools": {
                            "mode": "client",
                            "validator": "client-tools",
                        },
                        "targets": [{"step": "$return"}],
                    }
                ],
            }
        },
        "default_flow": "code",
    }


def _pid_path(config_path: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else config_path.parent / path


def _serve(
    config, pid_path: Path, config_path: Path | None = None
) -> None:
    family = socket.AF_INET6 if ":" in config.server.host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((config.server.host, config.server.port))
        sock.listen(2048)
    except OSError as exc:
        sock.close()
        raise SystemExit(
            f"cannot bind {config.server.host}:{config.server.port}: {exc}"
        ) from exc

    metadata = {
        "pid": os.getpid(),
        "host": config.server.host,
        "port": config.server.port,
        "started_at": time.time(),
    }
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pid_path.with_suffix(pid_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata), encoding="utf-8")
    temporary.replace(pid_path)
    try:
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(config, config_path=config_path),
                host=config.server.host,
                port=config.server.port,
            )
        )
        server.run(sockets=[sock])
    finally:
        sock.close()
        try:
            current = json.loads(pid_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        if current.get("pid") == os.getpid():
            pid_path.unlink(missing_ok=True)


def _status(pid_path: Path) -> int:
    try:
        metadata = json.loads(pid_path.read_text(encoding="utf-8"))
        pid = int(metadata["pid"])
        host = str(metadata["host"])
        port = int(metadata["port"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        print("stopped: no valid PID file")
        return 1
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"stale: PID {pid} is not running")
        return 1
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    try:
        response = httpx.get(f"http://{probe_host}:{port}/health", timeout=2)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"unhealthy: PID {pid} owns the service but health failed: {exc}")
        return 1
    print(f"running: PID {pid} at {host}:{port}")
    return 0


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
        configured = (
            f"{len(config.flows)} flow(s)"
            if config.uses_flows
            else f"{len(config.profiles)} profile(s)"
        )
        print(f"valid: {len(config.providers)} provider(s), {configured}")
        return

    if args.command == "serve":
        _serve(config, _pid_path(path, config.server.pid_file), path)
        return


    if args.command == "status":
        raise SystemExit(_status(_pid_path(path, config.server.pid_file)))

    if args.command == "benchmark":
        if not args.allow_code_execution:
            raise SystemExit("benchmark requires --allow-code-execution")
        payload = asyncio.run(
            run_benchmark(config, args.profiles, args.runs, args.output)
        )
        print(
            json.dumps(
                {
                    "summary": payload["summary"],
                    "strategy_gate": payload["strategy_gate"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
