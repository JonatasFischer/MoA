---
name: moa-operations
description: Use when operating the local MoA Gateway service, including start, stop, restart, status, health checks, model discovery, configuration validation, logs, launchctl, or port 14598 diagnostics.
---

# MoA Gateway Operations

Operate the local MoA Gateway from the repository root. Prefer the supervised
LaunchAgent lifecycle over managing individual Python processes.

## Service Facts

- Repository: `/Users/jonatas/sources/MoA`
- Configuration: `moa.yaml`
- Address: `http://127.0.0.1:14598`
- LaunchAgent label: `local.moa.gateway`
- LaunchAgent file: `~/Library/LaunchAgents/local.moa.gateway.plist`
- Standard log: `~/Library/Logs/moa-gateway.log`
- Error log: `~/Library/Logs/moa-gateway-error.log`
- Trace: `moa-trace.jsonl`

The LaunchAgent uses `KeepAlive`. Killing its current PID causes `launchd` to
spawn another process and is not a reliable restart procedure.

## Safe Restart

Before restarting after code or configuration changes:

```bash
uv run moa config validate
uv run pytest
```

Restart the loaded LaunchAgent:

```bash
launchctl kickstart -k gui/$(id -u)/local.moa.gateway
```

Wait briefly, then verify the process and public endpoints:

```bash
sleep 1
uv run moa status
curl -fsS http://127.0.0.1:14598/health
curl -fsS http://127.0.0.1:14598/v1/models
```

A successful restart must produce a running PID, `{"status":"ok"}`, and the
expected model aliases. Do not report success based only on `launchctl` returning
zero.

## Status And Diagnostics

Check the gateway's PID file and health endpoint:

```bash
uv run moa status
curl -fsS http://127.0.0.1:14598/health
```

Inspect the LaunchAgent when the process repeatedly exits or respawns:

```bash
launchctl print gui/$(id -u)/local.moa.gateway
```

Check the port owner when startup reports an address conflict:

```bash
lsof -nP -iTCP:14598 -sTCP:LISTEN
```

Read only the relevant log window. Gateway and trace logs can contain complete
prompts, tool results, and model output, so treat them as sensitive:

```bash
tail -n 100 ~/Library/Logs/moa-gateway-error.log
tail -n 100 ~/Library/Logs/moa-gateway.log
```

## Stop And Start

Use `bootout` for an intentional stop. Do not use repeated `kill` commands
against a KeepAlive service:

```bash
launchctl bootout gui/$(id -u)/local.moa.gateway
```

Load it again with the project LaunchAgent file:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.moa.gateway.plist
```

After loading, run the same status and endpoint checks used for a restart.

## Manual Foreground Mode

Use foreground mode only for debugging after the LaunchAgent has been stopped;
otherwise both processes will contend for port 14598:

```bash
uv run moa serve
```

Use `Ctrl-C` to stop foreground mode, then restore the LaunchAgent if normal
background operation is required.

## Configuration Changes

`PUT /api/config` applies valid flow configuration live and persists it
atomically. Code changes still require a process restart. Before any restart,
validate the persisted `moa.yaml` so a broken configuration does not enter a
KeepAlive crash loop.

## Safety Rules

- Never display or copy LaunchAgent environment-variable values; they may contain
  provider credentials.
- Never commit `.env`, trace files, PID files, or service logs.
- Do not modify the LaunchAgent file unless the user explicitly asks for service
  installation or environment changes.
- Do not commit, push, or restart merely because diagnostics were requested.
- Preserve unrelated worktree changes while operating the service.
