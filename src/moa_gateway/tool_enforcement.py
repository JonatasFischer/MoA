"""Aggregator-side enforcement for private investigation delegation."""

from __future__ import annotations

from moa_gateway.config import ToolEnforcementConfig


def get_aggregator_enforcement(
    enforcement: ToolEnforcementConfig, investigation_tool: str
) -> str:
    """Guide the aggregator to request only the investigation it needs."""

    if not enforcement.enabled:
        return ""

    if investigation_tool.startswith("stropha_rag_"):
        return f"""

# ADAPTIVE STROPHA INVESTIGATION

Call `{investigation_tool}` only when additional codebase information is needed to
answer the original request. Infer the task from the request and the specific
missing information; do not use a predetermined investigation scope. If the
available evidence is sufficient, answer without calling it. When calling it, emit
only the tool call on that turn, with no provisional reasoning or answer."""

    return f"""

# ADAPTIVE PRIVATE INVESTIGATION

You may call `{investigation_tool}` up to
{enforcement.max_investigation_calls} time(s) before answering. Create only the
investigations needed to obtain information missing from the conversation and the
advisory evidence. If the available information is sufficient, do not investigate.

## Delegation Requirements

1. Infer every investigation's scope from the original request and the specific
   information you still need. Do not use generic predetermined scopes.
2. When multiple independent questions remain, launch distinct research-only
   investigations in parallel, up to the configured maximum.
3. Every delegated prompt MUST require the investigator to use Stropha as its
   primary codebase source before reading files manually.
4. Ask investigators to return only a concise evidence-backed conclusion with
   file:line anchors, affected callers, relevant tests, and unresolved gaps.
5. Investigators must not edit files or execute the requested implementation.
6. Do not duplicate delegated work in the aggregator.

## Visibility Rule

If you call an investigation tool, emit tool calls only on that turn. Do not emit
reasoning, a provisional answer, implementation details, or a user-visible summary
before investigation results return. The gateway strips text emitted alongside an
investigation call.

After the tool results return, synthesize their conclusions with the contributor
evidence and continue the original task."""
