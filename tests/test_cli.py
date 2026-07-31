from __future__ import annotations

import json
import os
import socket

import httpx
import pytest

from moa_gateway.cli import _serve, _status
from moa_gateway.config import GatewayConfig


def _config(port: int) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {
                "host": "127.0.0.1",
                "port": port,
                "api_key_env": None,
            },
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://local.test/v1",
                }
            },
            "profiles": {
                "direct": {
                    "aliases": ["direct-code"],
                    "strategy": "direct",
                    "provider": "local",
                    "model": "model",
                }
            },
            "default_profile": "direct",
        }
    )


def test_serve_rejects_occupied_port(tmp_path) -> None:
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen()
    port = occupied.getsockname()[1]

    with pytest.raises(SystemExit, match="cannot bind"):
        _serve(_config(port), tmp_path / "moa.pid")

    occupied.close()


def test_serve_owns_and_cleans_pid_file(tmp_path, monkeypatch) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    pid_path = tmp_path / "moa.pid"

    def fake_run(self, sockets) -> None:
        metadata = json.loads(pid_path.read_text())
        assert metadata["pid"] == os.getpid()
        assert sockets[0].getsockname()[1] == port

    monkeypatch.setattr("uvicorn.Server.run", fake_run)

    _serve(_config(port), pid_path)

    assert not pid_path.exists()


def test_status_reports_stale_and_healthy_processes(tmp_path, monkeypatch) -> None:
    pid_path = tmp_path / "moa.pid"
    pid_path.write_text(
        json.dumps({"pid": 123, "host": "127.0.0.1", "port": 14598})
    )
    monkeypatch.setattr("os.kill", lambda pid, signal: (_ for _ in ()).throw(OSError()))
    assert _status(pid_path) == 1

    monkeypatch.setattr("os.kill", lambda pid, signal: None)
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", "http://test/health")
        ),
    )
    assert _status(pid_path) == 0
