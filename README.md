# MoA Gateway

A local, configurable Mixture-of-Agents gateway for Claude Code, OpenCode, and
Codex. The project is under active development. See [PLAN.md](PLAN.md) for the
architecture, protocol requirements, and milestones.

The current implementation is the first direct-mode compatibility slice. It
provides a YAML configuration, CLI, health and model-discovery endpoints, and
text translation for Anthropic Messages, OpenAI Chat Completions, and OpenAI
Responses over an Ollama/OpenAI-compatible upstream.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run moa config validate
uv run moa serve
```

The default configuration is in `moa.yaml`. Change its upstream model to one
available from `ollama list` before sending requests.

## Current Limitations

- Only the `direct` strategy is enabled.
- Client tool calling is rejected rather than silently discarded.
- Input is text-only.
- Token counting and full provider error translation are not implemented yet.

These limitations correspond to the staged implementation in `PLAN.md`.
