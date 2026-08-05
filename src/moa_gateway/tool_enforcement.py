"""Aggregator-side enforcement for private investigation delegation."""

from __future__ import annotations

from moa_gateway.config import ToolEnforcementConfig


def get_aggregator_enforcement(
    enforcement: ToolEnforcementConfig, investigation_tool: str
) -> str:
    """Require the aggregator to delegate investigation before answering."""

    if not enforcement.enabled:
        return ""

    if investigation_tool.startswith("stropha_rag_"):
        return f"""

# MANDATORY STROPHA INVESTIGATION

You are executing a private delegated investigation. Call
`{investigation_tool}` now using the user's investigation request as the task.
Emit only the tool call on this turn, with no provisional reasoning or answer.
After its result returns, provide the concise evidence-backed conclusion requested
by the parent agent. Do not edit files."""

    consequence = (
        f"The gateway rejects responses that omit `{investigation_tool}` and strips"
        if enforcement.enforcement_mode in {"auto", "block"}
        else "The gateway strips"
    )

    return f"""

# MANDATORY PRIVATE INVESTIGATION

Before answering, implementing, or exposing analysis to the user, call
`{investigation_tool}`. This delegated investigation is private working context.

## Delegation Requirements

1. Launch multiple independent research-only investigations in parallel when the
   request has separable architectural, implementation, and testing concerns.
2. Every delegated prompt MUST require the investigator to use Stropha as its
   primary codebase source before reading files manually.
3. Ask investigators to return only a concise evidence-backed conclusion with
   file:line anchors, affected callers, relevant tests, and unresolved gaps.
4. Investigators must not edit files or execute the requested implementation.
5. Do not duplicate delegated work in the aggregator.

## Visibility Rule

Emit tool calls only on this turn. Do not emit reasoning, a provisional answer,
implementation details, or a user-visible summary before investigation results
return. {consequence} any text emitted alongside the investigation call.

After the tool results return, synthesize their conclusions with the contributor
evidence and continue the original task."""
