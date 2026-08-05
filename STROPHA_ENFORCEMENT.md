# Private Investigation Enforcement

MoA keeps tool authority with the aggregator. Request filters and contributors
remain private advisory model calls and do not receive client tool definitions.

For the bundled `code` profile, `server.tool_enforcement` makes OpenCode's
`task` tool available for adaptive private investigation on the initial aggregation
turn:

```yaml
server:
  tool_enforcement:
    enabled: true
    investigation_tools: [task]
    max_investigation_calls: 3
```

The aggregator decides whether additional information is needed and may launch up
to the configured maximum of independent research-only tasks. It infers every task
scope from the original request and the missing evidence; zero tasks are valid when
the available context is sufficient. MoA does not execute the tool. It returns the
grounded calls to OpenCode, which runs the private investigators and adds their
results to the conversation.

MoA normalizes delegated calls to use OpenCode's read-only `explore` subagent,
grounds each call with the original request, adds the Stropha contract when the
model omits it, and caps calls at the configured maximum. MoA never expands a call
or inserts generic architecture, implementation, or verification scopes. When a
delegated request enters MoA, the contract marker routes it through the direct tool
dispatcher, preventing recursive task delegation while keeping tool execution in
OpenCode.

Some OpenCode subagent configurations expose only basic `glob`/`grep`/`read`
tools. A delegated request falls back to those available research tools when no
Stropha tool is present; it is traced as a degraded investigation instead of
failing the entire parent session.

## Adaptive Behavior

MoA preserves the client's `tool_choice`; it does not force a named tool. If none
of the configured investigation tools are available, aggregation continues without
the adaptive instruction. During streaming, aggregator output is buffered until it
is known whether an investigation was requested. Text emitted alongside an
investigation call is discarded so provisional reasoning does not leak to the
coding agent.

Tool-result turns continue through the configured direct tool dispatcher. This
avoids recursively running another council while the aggregator consumes the
private investigator's conclusion.
