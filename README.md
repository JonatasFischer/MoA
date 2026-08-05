# MoA Gateway

A local, configurable Mixture-of-Agents gateway for Claude Code, OpenCode, and
Codex. The project is under active development. See [PLAN.md](PLAN.md) for the
architecture, protocol requirements, and milestones.

The current implementation provides direct, classic, and council MoA strategies,
a typed YAML configuration, CLI, health and model-discovery endpoints, and text
translation for Anthropic Messages, OpenAI Chat Completions, and OpenAI
Responses over native Ollama, OpenAI, and DeepSeek upstreams.

The default `code` profile asks three model families to each run the complete
five-perspective council: `qwen2.5-coder:7b`, `gemma4:latest`, and
`deepseek-coder-v2:16b`. Each
contributor returns schema-validated Contrarian,
Software Architect, Clean Coder, Pragmatic Engineer, and Engineering Manager
fields. Before querying them, the aggregator model runs one request-analysis filter;
its output becomes additional untrusted context for every contributor and the final
aggregation. Aggregation starts after two valid responses or fails at the 45-second
deadline; the aggregator acts as the implementing Engineer, considers every
perspective, controls scope and complexity, and proceeds with the smallest correct
implementation instead of returning a council transcript.

## Development

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen3.6:27b
ollama pull gemma4
ollama pull deepseek-coder-v2:16b
uv sync --all-groups
uv run pytest
uv run moa config validate
uv run moa serve
uv run moa status
```

The default configuration is in `moa.yaml`. Public model aliases are
`claude-moa-code`, `moa-code`, `claude-direct-code`, `direct-code`,
and `deepseek-direct-code`.

## Flow Lab

After starting the gateway, open `http://127.0.0.1:14598/`. Flow Lab is an
unauthenticated local control surface for experimenting with direct, classic,
and council structures. It can create and duplicate flows, arrange contributor,
aggregator, and tool-dispatch targets, manage provider connections, and discover
the model IDs exposed by a provider.

The diagram follows the runtime branches implemented in `Gateway`: profile and
continuation routing, tool-enforcement preflight, the aggregator-backed request
filter, parallel contribution quorum, aggregation and output enforcement,
empty-output retry and fallback, and the conditional tool-dispatch bypass. It
also marks semantic refinement as not implemented rather than presenting
recovery as a critique/revision pass.

Enter a prompt in **Run the selected flow** to execute that profile against the
real configured providers. The diagram updates from request-scoped gateway trace
events, showing stage progress, parallel quorum state, cancellations, retries,
failures, and completion. Select a node to inspect its raw model output, tool
calls, usage, errors, and trace records. Runs consume normal provider tokens and
can be stopped from the dashboard. If tool enforcement produces a client-side
tool call, the run stops at that real boundary; the dashboard does not fabricate
the external tool result.

`Apply live` validates the complete experiment before changing anything. New
requests use the new configuration immediately; requests already in progress
finish on the configuration generation with which they started. When MoA is
started with `moa serve`, accepted changes are also written atomically to the
selected `--config` YAML file. Apps created programmatically without a config
path apply changes only for the lifetime of the process.

The Flow Lab and its `/api/config` control API intentionally have no
authentication. Keep the gateway bound to loopback unless every user with
network access should be allowed to inspect and replace the active experiment.

## OpenAI And DeepSeek Compatibility

Start the gateway with `uv run moa serve`. It accepts Chat Completions at both
`http://127.0.0.1:14598/v1/chat/completions` (OpenAI-style base URL) and
`http://127.0.0.1:14598/chat/completions` (DeepSeek-style base URL). The same
request and response schema, bearer authentication, streaming, and function
tools are supported on both paths.

For OpenAI-compatible clients, set the base URL to
`http://127.0.0.1:14598/v1` and select `moa-code`:

```bash
curl http://127.0.0.1:14598/v1/chat/completions \
  -H "Authorization: Bearer $MOA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"moa-code","messages":[{"role":"user","content":"Hello"}]}'
```

OpenAI and DeepSeek are first-class MoA backend adapters over their compatible
Chat Completions APIs. Their standard URLs and API-key environment variable
names are defaults, so only the provider type is required:

```yaml
providers:
  openai:
    type: openai
    timeout_seconds: 1800
  deepseek:
    type: deepseek
    timeout_seconds: 1800
```

Set `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` before starting the gateway. A
custom `base_url` or `api_key_env` can still override either default.

Any profile target can select these providers. Council profiles require at
least two contributors with distinct `family` values. Every contributor runs
all five perspectives, and the aggregator receives every complete answer:

```yaml
profiles:
  code:
    aliases: [moa-code]
    strategy: council
    contributors:
      - provider: ollama
        model: gemma4:latest
        family: gemma
      - provider: ollama
        model: deepseek-coder-v2:16b
        family: deepseek
    aggregator:
      provider: ollama
      model: qwen3.6:27b
      family: qwen
      think: false
    min_quorum: 2
    max_concurrency: 1
    contributor_deadline_seconds: 45
    contributor_max_tokens: 2560
    contributor_format: json-schema
    reasoning_reserve:
      qwen: 4096
```

For aggregation, a configured family reserve is added to the client's output
budget before the upstream call. If aggregation still returns no content or
tool calls, MoA retries once with twice that internal budget and thinking
disabled, then falls back to the strongest non-empty contributor response.
MoA returns an upstream error rather than an empty successful response when no
fallback is available.

Native Ollama targets can also set `keep_alive` and `num_ctx`. The bundled
configuration warms the active `code` profile at startup and records Ollama's
load, prompt-evaluation, and generation durations. Tool-result turns bypass the
council and route directly to the configured tool dispatcher. Contributor input
excludes the coding-agent system prompt, tool definitions, and replayed tool
results.

Before the initial `code` aggregation, MoA requires the client-provided `task`
tool and instructs the aggregator to delegate private, Stropha-backed
investigations. MoA validates the tool call but does not execute it; OpenCode runs
three focused investigators in parallel and returns only their conclusions. Any
text emitted beside the required call is suppressed so pre-investigation
reasoning is not exposed to the coding agent. See
[STROPHA_ENFORCEMENT.md](STROPHA_ENFORCEMENT.md).

The bundled `moa.yaml` also includes a `deepseek-direct-code` profile. The
DeepSeek adapter can be selected by any custom contributor or aggregator.

Streaming clients receive SSE progress comments while the contributor quorum and
aggregation stages run. The local OpenCode provider still uses 30-minute
full-request, header, and stream chunk timeouts for model loading and long
generations.

## Execution Trace

The bundled configuration writes an append-only JSONL trace to
`moa-trace.jsonl`. Records are flushed immediately and correlated by
`request_id`. They include model inputs, complete contributor and aggregator
outputs, failures, cancellations, stage lifecycle, provider timing breakdowns,
per-stage and total token usage, contributor scores, tool calls, and streaming
deltas. `X-MoA-Parent-Request-ID` can correlate nested agent work; every API
response returns `X-MoA-Request-ID`.

Follow all activity in real time:

```bash
tail -f moa-trace.jsonl
```

With `jq`, show only completed model responses:

```bash
tail -f moa-trace.jsonl | jq -c \
  'select(.event == "model_completed") | {request_id, stage, model, duration_seconds, content}'
```

Set `server.trace_log_path: null` to disable tracing. The trace contains full
prompts, tool results, and model responses, so treat it as sensitive local data.
The default 32 MiB cap retains three rotated backups.

## Benchmark

Run the executable coding benchmark with:

```bash
uv run moa benchmark --runs 3 \
  --output benchmark-results.json --allow-code-execution
```

Generated Python is AST-screened before running deterministic tests in a
temporary directory. The default arms are `direct`, `council-k2`, `council-k3`,
and `self-consistency`; promotion requires a strict pass@1 improvement over
direct. See [BENCHMARK.md](BENCHMARK.md) for details.

## Current Limitations

- Anthropic Messages and OpenAI Responses tool calling is rejected rather than
  silently discarded; Chat Completions function tools are supported.
- Input is text-only.
- Client prompt usage is a deterministic estimate; trace `usage_total` uses the
  exact token counts reported by every provider stage.

These limitations correspond to the staged implementation in `PLAN.md`.
