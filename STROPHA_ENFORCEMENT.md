# Private Investigation Enforcement

MoA keeps tool authority with the aggregator. Request filters and contributors
remain private advisory model calls and do not receive client tool definitions.

For the bundled `code` profile, `server.tool_enforcement` requires OpenCode's
`task` tool on the initial aggregation turn:

```yaml
server:
  tool_enforcement:
    enabled: true
    required_tools: [task]
    enforcement_mode: auto
    min_investigation_calls: 3
```

The aggregator is instructed to launch independent research-only tasks, require
those investigators to use Stropha, and request concise conclusions with source
anchors, callers, tests, and unresolved gaps. MoA does not execute the tool. It
returns the validated call to OpenCode, which runs the private investigator and
adds only its final result to the conversation.

MoA normalizes delegated calls to use OpenCode's read-only `explore` subagent,
adds the Stropha contract when the model omits it, and expands a single call into
the configured minimum parallel investigations. The bundled configuration uses
architecture, implementation/callers, and verification/risks focuses. When each
delegated request enters MoA, the contract marker switches enforcement from
`task` to the first available Stropha investigation tool. This prevents recursive
task delegation while still keeping every tool execution in OpenCode.

Some OpenCode subagent configurations expose only basic `glob`/`grep`/`read`
tools. A delegated request falls back to those available research tools when no
Stropha tool is present; it is traced as a degraded investigation instead of
failing the entire parent session.

## Enforcement Modes

- `warn`: add the delegation instruction when a configured tool is available.
- `block`: require a configured tool call and reject a response that omits it.
- `auto`: force the first available configured tool with named `tool_choice` and
  reject a response that omits it.

For `block` and `auto`, MoA rejects the request before running the filter or
contributors when none of the configured tools is available. During streaming,
the aggregator response is buffered until the required call is validated. Text
emitted alongside that call is discarded so provisional reasoning does not leak
to the coding agent.

Tool-result turns continue through the configured direct tool dispatcher. This
avoids recursively running another council while the aggregator consumes the
private investigator's conclusion.
