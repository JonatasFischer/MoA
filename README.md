# MoA Gateway

A local, configurable Mixture-of-Agents gateway for Claude Code, OpenCode, and
Codex. MoA exposes configured flows as model IDs while executing each flow as a
validated graph of AI steps and synchronization gates.

The gateway accepts Anthropic Messages, OpenAI Chat Completions, and OpenAI
Responses requests over native Ollama, OpenAI, DeepSeek, and generic
OpenAI-compatible providers.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run moa config validate
uv run moa serve
uv run moa status
```

The active configuration is `moa.yaml`. The bundled public model aliases are:

- `claude-direct-code`
- `direct-code`
- `moa-code`
- `benchmark-council-k2`
- `benchmark-council-k3`
- `benchmark-self-consistency`
- `deepseek-direct-code`

## Configured Flows

Configuration version 2 has four primary catalogs:

- `providers`: upstream connections and credentials.
- `prompts`: reusable system and context templates.
- `tool_validators`: deterministic validation and transformation of model tool calls.
- `flows`: public models composed from AI steps and gates.

Providers declare accepted input modalities globally or per model. Text is always
required and is the safe default:

```yaml
providers:
  local:
    type: openai-compatible
    base_url: http://127.0.0.1:11434/v1
    input_modalities: [text]
    model_input_modalities:
      vision-model: [text, image]
      document-model: [text, image, file]
```

OpenAI image/file blocks, Anthropic image/document blocks, and Responses
`input_image`/`input_file` blocks are normalized into one canonical representation.
Requests are rejected before provider transport when the selected model does not
declare every required modality.

Every flow is stored as a list of steps. Links between step IDs define the actual
execution graph; list order is only for readability.

Flows can opt into deterministic request-shape routing with a `simple_request`
start. Existing continuation starts should remain first, and `always` remains the
required fallback:

```yaml
routing:
  max_latest_user_chars: 800
  max_conversation_chars: 4000
  max_messages: 4
  require_no_tools: true
starts:
- {step: continue, when: tool_continuation, priority: 10}
- {step: direct-answer, when: simple_request, priority: 20}
- {step: request-filter, when: always, priority: 100}
```

Start routes form a chain of responsibility. Lower priority numbers are evaluated
first and the first matching route wins. When priority is omitted, declaration
order is retained for compatibility.

### AI Steps

An `ai` step selects its prompt, provider, model, conversation visibility,
generation controls, tools, validation, retries, fallback, and targets:

```yaml
- id: reviewer
  type: ai
  prompt: classic-proposer
  prompt_variables:
    role: independent reviewer
  provider: ollama
  model: gemma4:latest
  conversation: advisory
  max_tokens: 1024
  temperature: 0.2
  targets:
  - step: proposals
```

Prompt templates can use:

- `{{request}}`: latest user request.
- `{{conversation}}`: selected canonical conversation as JSON.
- `{{tools}}`: client tool names and descriptions.
- `{{inputs}}`: results that activated the current step.
- `{{inputs_full}}`: activating completions with text, tool calls, finish reason, and model.
- `{{steps.<id>}}`: completed output from a named prior step.
- `{{available_skills}}`: skill names and descriptions advertised by the client.
- `{{loaded_skills}}`: completed `skill` tool results in the conversation.
- `{{investigation_results}}`: tool-result content already in the conversation.
- `{{remaining_investigations}}`: remaining allowance on the current step.

### Gates

A `gate` is notified by every step that targets it. It bounds concurrency, applies
one shared deadline, waits for all sources or the deadline, and requires at least
`min_success` successful results:

```yaml
- id: contributions
  type: gate
  min_success: 2
  max_concurrency: 3
  deadline_seconds: 45
  completion: all-or-deadline
  on_failure: fail
  targets:
  - step: aggregate
```

The compiler rejects cycles, duplicate links, conditional gate sources, unreachable
steps, gates without enough sources, unknown references, invalid JSON Schemas,
fallback gates that are not ancestors, and any path that cannot reach `$return`.

### Public Models

Aliases on a flow become entries in `GET /v1/models`:

```yaml
flows:
  direct:
    aliases: [claude-direct-code, direct-code]
    starts:
    - {step: answer, when: always}
    output: {step: answer}
    steps:
    - id: answer
      type: ai
      provider: lms
      model: Qwen/Qwen3-Coder-Next-FP8
      conversation: full
      tools:
        mode: client
        validator: client-tools
      targets:
      - step: $return

default_flow: code
```

Requests can select either an internal flow name or one of its aliases. Discovery
advertises aliases.

## Default Code Flow

The bundled `code` flow is entirely represented in `moa.yaml`:

```text
request-filter
  -> solution-council ------------+
  -> architecture-council --------+-> contributions gate -> aggregate
  -> project-patterns-council -----+                            |
                                                  action-skill-reinforcement
                                                    | approved -> return proposal unchanged
                                                    | missing skill -> skill tool call
                                                    | evidence gap -> task tool call

skill result -> request-filter (rerun the complete panel with loaded guidance)
other tool result -> integrate-investigation -> action-skill-reinforcement
```

The contributor gate requires all three specialized responses, permits three concurrent
calls, and uses a shared 45-second deadline. Structured council responses are
validated against `council-response` and repaired once when invalid. Empty
aggregation retries once with twice the token budget and thinking disabled, then
falls back to the best non-empty gate result.

## Action And Skill Reinforcement

The aggregator proposes an answer or action but cannot return it directly. A dedicated
terminal reinforcement step receives the complete proposal, the client skill catalog,
loaded skill results, and previous tool evidence.

When an available skill governs the proposed action and has not been loaded, only a
`skill` call is returned. Its result starts a new run at `request-filter`, so the panel
and aggregator regenerate the proposal with the loaded guidance. Skill names are
validated against the client's `<available_skills>` catalog and already-loaded skills
cannot be requested again.

After skill requirements are satisfied, the same step decides whether a material
unanswered question could change correctness or implementation. When no reinforcement
is needed, its private approval is discarded and the proposed completion is returned
unchanged, preserving tool IDs and arguments. The configured validator:

- requires the tool to exist in the client request;
- requires a stable call ID and JSON-object arguments;
- forces `subagent_type: explore` for `task`;
- prepends the latest user request;
- always appends the mandatory Stropha evidence contract;
- validates requested skill names against the advertised catalog;
- removes private text emitted beside the investigation call;
- never executes the tool inside MoA.

The client executes reinforcement tools. `task` results start at
`integrate-investigation`; `skill` results rerun the main panel. The maximum combined
private reinforcement calls is mandatory configuration on the output step.

Normal tool results use the same continuation path, so coding-agent loops do not
rerun the contributor panel on every read, command, or edit result.

## Flow Lab

Start the gateway and open `http://127.0.0.1:14598/`.

Flow Lab reads the same v2 flow definitions that the core compiles. It renders every
configured step and target as a graph, edits AI and gate properties, manages prompts
and providers, validates changes, and writes accepted changes atomically to the
selected YAML file.

Runs use real providers and tokens. If a draft has unapplied changes, Flow Lab first
applies and validates it so the displayed graph cannot differ from the executed
generation. Trace events carry `flow_id` and `node_id`, allowing runtime state and
outputs to be associated directly with graph nodes.

Requests already in progress retain the configuration generation with which they
started. New requests use the replacement generation.

The Flow Lab and `/api/config` control API are intentionally unauthenticated. Keep
the gateway bound to loopback unless every network user should be able to inspect
and replace the active configuration.

## OpenAI Compatibility

Start the gateway with `uv run moa serve`, then use the OpenAI-compatible base URL
`http://127.0.0.1:14598/v1`:

```bash
curl http://127.0.0.1:14598/v1/chat/completions \
  -H "Authorization: Bearer $MOA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"moa-code","messages":[{"role":"user","content":"Hello"}]}'
```

The unversioned DeepSeek-style endpoint is also available at
`/chat/completions`.

## Live Reconfiguration

`PUT /api/config` validates and compiles every flow before activation. Invalid
updates leave the current generation untouched. Accepted updates are persisted by
writing a temporary YAML file and atomically replacing the configured path.

Provider instances and old flow snapshots remain available to requests holding a
runtime lease. They are closed after the final request on that generation completes.

## Execution Trace

The bundled configuration writes append-only JSONL to `moa-trace.jsonl`. Records
include request IDs, parent request IDs, flow and node IDs, model inputs and outputs,
gate progress, failures, retries, tool validation, provider timing, and usage.

```bash
tail -f moa-trace.jsonl
```

Trace files contain full prompts, tool results, and model responses. Treat them as
sensitive local data.

## Benchmark

Run the coding benchmark with:

```bash
uv run moa benchmark --runs 3 \
  --output benchmark-results.json --allow-code-execution
```

The default arms are `direct`, `council-k2`, `council-k3`, and
`self-consistency`. They are ordinary configured flows, not hard-coded runtime
strategies.

## Current Limitations

- Function tools, calls, results, and argument deltas are supported across Chat
  Completions, Anthropic Messages, and OpenAI Responses. Responses custom tools,
  local-shell items, and server-managed conversations remain unsupported.
- Image and file input is capability-gated. The bundled coding models remain
  text-only until a compatible provider/model is explicitly configured. Native
  Ollama image input accepts the image values supported by its chat API; files need
  an OpenAI-compatible provider.
- Terminal AI steps stream provider deltas directly. A step that still requires
  schema repair, transformed/discarded tool output, or a post-answer checker remains
  buffered because already emitted protocol events cannot be retracted.
- Client prompt usage falls back to a deterministic estimate only when the final
  provider omits counts. Trace usage totals include every provider call, retry, and
  repair.
