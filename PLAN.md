# Configurable Mixture-of-Agents Gateway Plan

## Goal

Build a local gateway that appears as one model to Claude Code, OpenCode, and
Codex while running a configurable Mixture-of-Agents (MoA) pipeline against
local models. The first upstream target is Ollama. The service is implemented
in Python with FastAPI and configured through YAML plus a CLI.

The public API must be a real agent-compatible API, not only a text-compatible
chat endpoint. Tool calls, tool results, streaming event order, stop reasons,
usage, and errors must remain valid for each client protocol.

## Research Basis

Together AI proposed the original layered MoA architecture:

- Repository: <https://github.com/togethercomputer/MoA>
- Paper: <https://arxiv.org/abs/2406.04692>

The original implementation fans a request out to multiple proposers, gathers
their complete answers, and asks an aggregator to synthesize a final response.
It also demonstrates additional refinement layers. The paper reports that
synthesis performs better than merely selecting one candidate, that proposer
quality and diversity both matter, and that the first aggregation layer gives
the largest quality improvement. Its primary operational limitation is high
time to first token because final generation cannot begin until the preceding
layer completes.

Later Self-MoA research (<https://arxiv.org/abs/2502.00674>) found that repeated
sampling from one strong model can outperform a mixture containing weaker
models. Consequently, this gateway must treat model mixtures as benchmarked
configuration rather than assuming that greater model diversity always helps.

Existing MoA gateways are useful references but do not meet this project's
agent requirements. Some expose only OpenAI Chat Completions, omit native
Anthropic Messages or OpenAI Responses, or explicitly drop client tool calls.
A coding client can appear connected in those systems while never executing a
file edit or command.

## Chosen Product Direction

- Deployment: local, single user
- Runtime: Python 3.12 and FastAPI
- Initial upstream: Ollama through its OpenAI-compatible interface
- Configuration: YAML with environment-variable secret references
- Configuration assistance: CLI initialization and validation commands
- Default target strategy: classic mixed-model MoA on every turn
- Tool authority: only the final aggregator may emit a client-visible action
- Initial development sequence: protocol-compatible direct mode before MoA

## API Surfaces

| Client | Required surface |
| --- | --- |
| Claude Code | `POST /v1/messages`, Anthropic SSE, optional `POST /v1/messages/count_tokens`, and model discovery |
| OpenCode | `POST /v1/chat/completions`, `POST /v1/responses`, or the Anthropic surface |
| Codex | `POST /v1/responses`; custom Codex providers use the Responses wire API |
| All clients | `GET /v1/models`, bearer/API-key authentication, native errors, cancellation, and tool loops |

Claude Code may request `/v1/messages?beta=true`, so routing must match the
path independently of the query string. Its inference response must stream
without buffering the final model response. The gateway must accept both
`Authorization: Bearer` and `x-api-key` credentials.

Each facade has its own event grammar:

- Anthropic: `message_start`, content-block lifecycle events,
  `message_delta`, and `message_stop`.
- Chat Completions: `chat.completion.chunk` objects followed by `[DONE]`.
- Responses: response, output-item, content-part, text/tool delta, and
  completion lifecycle events.

The three protocols are adapters over one canonical conversation model, not
aliases for the same JSON endpoint.

## Strategy Catalog

Strategies are composable topology, fusion, and control policies.

| Strategy | Operation | Primary use |
| --- | --- | --- |
| Direct | One model answers | Baseline and low latency |
| Classic mixed MoA | Different proposers run in parallel and one model synthesizes | Default target |
| Self-MoA | One strong model is sampled repeatedly | Avoid weak-panel degradation |
| Role-specialized MoA | Proposers receive coder, reviewer, tester, security, or API roles | Coding tasks |
| Multi-layer MoA | Each layer sees all outputs from the prior layer | Maximum quality at high cost |
| Sequential MoA | Candidates are incorporated through a sliding window | Limited context windows |
| Compose | Merge complementary candidate sections | Research and design |
| Solve-first synthesis | Aggregator solves independently, then audits proposals | Reduce anchoring |
| Judge then synthesize | A structured judge precedes final generation | Conflicting technical answers |
| Rank/select | Return one selected candidate unchanged | Lower synthesis cost |
| Best-of-N | Score repeated candidates and select one | Verifiable code or math |
| Majority vote | Select the most common answer | Classification |
| Weighted consensus | Weight candidates by measured quality | Stable benchmarked panels |
| Debate | Agents critique each other over rounds | Ambiguous reasoning |
| Critique/revise | Draft, critique, rewrite | Small sequential panels |
| Planner/worker | Decompose tasks and integrate worker outputs | Large tasks |
| Cascade | Start cheap and escalate when necessary | Cost control |
| Sparse MoA | Activate only relevant specialists | Large model catalogs |
| Adaptive routing | Select direct, light, or full MoA by complexity | Later optimization |
| Early exit | Stop when agreement or verification is sufficient | Latency reduction |
| Quorum | Continue after enough proposers succeed | Failure tolerance |
| Semantic/attention fusion | Rank candidate sections before synthesis | Large candidate sets |
| Verifier-guided | Tests or schemas score candidates | Code and structured output |
| Chunk-wise aggregation | Aggregate partial streams | Experimental TTFT reduction |

The initial strategy interface will support `direct`, `classic`, `self`,
`judge_synthesize`, and `sequential`. Only `direct` is enabled during the first
protocol-conformance milestone.

## Coding-Agent Tool Policy

Merging independently generated tool calls is unsafe. The gateway uses one
authoritative action source:

1. Normalize the client request and tool definitions.
2. Give proposers the conversation and an advisory description of tools.
3. Capture proposer text or proposed actions without executing or returning
   them.
4. Give the final aggregator the original client tool definitions and the
   candidate responses.
5. Return only the aggregator's native text or tool calls.
6. Let Claude Code, OpenCode, or Codex apply its normal permission and execution
   policy.
7. Normalize the client's tool result on the next turn and repeat the pipeline.

Tool identifiers must remain stable across protocol translation:

- Anthropic `tool_use.id`
- Chat Completions `tool_calls[].id`
- Responses `call_id`

Proposer actions never reach the client. This prevents duplicate edits,
conflicting shell commands, and bypasses of client-side permission systems.

## Architecture

```text
Claude Code ---- Anthropic Messages ----+
OpenCode ------- Chat/Responses --------+--> Protocol codecs
Codex ---------- OpenAI Responses ------+
                                               |
                                      Canonical request IR
                                               |
                                      Profile/strategy registry
                                               |
                                         MoA orchestrator
                                      /        |         \
                               proposers     judge     aggregator
                                      \        |         /
                                      Provider adapters
                                               |
                                      Ollama/OpenAI-compatible
```

### Modules

- `protocols`: request parsing, native response formatting, SSE, and errors
- `domain`: canonical content blocks, messages, tools, usage, and stop reasons
- `strategies`: direct, classic, self, sequential, and judge implementations
- `providers`: Ollama/OpenAI-compatible provider abstraction
- `orchestrator`: concurrency, deadlines, cancellation, retries, and quorum
- `config`: YAML loading, cross-reference validation, and secret resolution
- `cli`: setup, validation, model probing, diagnostics, and serving
- `observability`: structured logs, per-stage latency, usage, and failures

Unknown request fields must not crash parsing. Anthropic beta headers and
fields are open-ended and change as Claude Code evolves. Translation to a
non-Anthropic upstream must either implement a capability or reject it clearly;
it must never silently drop client actions.

## Configuration Shape

```yaml
server:
  host: 127.0.0.1
  port: 8080
  api_key_env: MOA_API_KEY

providers:
  local:
    type: openai-compatible
    base_url: http://127.0.0.1:11434/v1

profiles:
  code:
    aliases:
      - claude-moa-code
      - moa-code
    strategy: classic
    proposers:
      - provider: local
        model: proposer-model-a
      - provider: local
        model: proposer-model-b
    aggregator:
      provider: local
      model: tool-capable-aggregator
    tool_policy: final-only
    min_quorum: 1
    max_concurrency: 2
    reference_token_budget: 12000

default_profile: code
```

`claude-moa-code` begins with `claude`, which allows Claude Code gateway model
discovery to include it. Several aliases may resolve to one profile.

## Implementation Milestones

### 1. Foundation

- Create the typed Python project and CLI.
- Load and validate YAML configuration.
- Add health, startup, and model discovery endpoints.
- Add an async Ollama/OpenAI-compatible provider.
- Add offline tests and a Docker-ready entry point.

### 2. Protocol-Compatible Direct Mode

- Implement text requests and non-streaming responses for all three facades.
- Implement exact text SSE lifecycle events for all three facades.
- Normalize messages, controls, usage, stop reasons, and upstream errors.
- Verify official OpenAI and Anthropic SDK behavior.
- Verify real Claude Code, OpenCode, and Codex text conversations.

### 3. Agent Tool Conformance

- Normalize function definitions, calls, and results.
- Support parallel tool calls and fragmented JSON argument streams.
- Map Codex function, custom, local-shell, and apply-patch item types.
- Preserve Responses assistant phases and required reasoning items.
- Verify file read, command, edit, patch, and continuation loops in each client.

### 4. Classic MoA

- Add advisory proposer fan-out.
- Add failure quorum, deadlines, and cancellation.
- Add candidate anonymization and reference token limits.
- Add hardened synthesis and final-only tool authority.
- Stream the final aggregator in the native client protocol.

### 5. Configuration and Diagnostics

- Add `moa init`, `moa config validate`, `moa models probe`, and client doctors.
- Add YAML reload with atomic validation.
- Add per-stage logs and `x-moa-*` diagnostics.
- Record aggregate provider usage separately from protocol context usage.

### 6. Additional Strategies and Evaluation

- Add Self-MoA, sequential, judge/synthesis, roles, routing, and early exit.
- Benchmark solo and MoA on identical coding tasks.
- Measure tool-call validity, patch correctness, test success, latency, token
  use, and failure recovery rather than relying only on prose benchmarks.

## Usage Accounting

Standard response usage must describe the final model context and output in the
facade's native semantics. Summing duplicated proposer input into Anthropic's
`input_tokens` could cause Claude Code to compact its conversation too early.

Total panel usage and cost are operational metadata and will be recorded in
structured logs and response headers. Later observability endpoints may expose
per-request stage details.

## Acceptance Criteria

- Official OpenAI SDKs work against Chat Completions and Responses.
- Official Anthropic SDKs work against Messages and streaming.
- Claude Code executes and continues after tool calls through the gateway.
- OpenCode completes equivalent loops through OpenAI and Anthropic facades.
- Codex completes Responses tool calls and result continuations.
- Randomly fragmented SSE tool arguments reconstruct correctly.
- A proposer failure is tolerated when quorum is met.
- Client disconnect cancels outstanding provider requests.
- Unknown optional fields do not crash request parsing.
- The default MoA profile is evaluated against a direct baseline.

## Risks

- Ollama may serialize fan-out or swap models when available memory is too low.
- Classic MoA delays the first final token until proposers complete.
- Weak proposers can reduce final quality.
- The final local model must already be a reliable tool caller.
- Claude Code and Codex protocol capabilities evolve frequently.
- Additional MoA layers multiply context, latency, and memory pressure.

The first release should therefore prove direct protocol and tool compatibility
before enabling classic MoA by default.
