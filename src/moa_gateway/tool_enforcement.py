"""Tool enforcement for MCP server usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolEnforcementConfig:
    """Configuration for tool enforcement."""
    
    enabled: bool = False
    required_tools: list[str] = None
    enforcement_mode: Literal["warn", "block", "auto"] = "warn"
    
    def __post_init__(self):
        if self.required_tools is None:
            self.required_tools = []


def get_stropha_enforcement_prompt(enforcement: ToolEnforcementConfig) -> str:
    """Get the enforcement prompt for Stropha MCP usage."""
    
    if not enforcement.enabled:
        return ""
    
    mode_messages = {
        "warn": "You MUST use Stropha MCP tools for codebase exploration. Failure to do so will result in incomplete analysis.",
        "block": "You MUST use Stropha MCP tools for codebase exploration. Your response will be rejected if you don't use them.",
        "auto": "You MUST use Stropha MCP tools for codebase exploration. This is required for all codebase-related queries.",
    }
    
    required = ", ".join(f"`{tool}`" for tool in enforcement.required_tools)
    
    return f"""# MANDATORY TOOL USAGE

{mode_messages[enforcement.enforcement_mode]}

## Required Tools

You MUST use the following Stropha MCP tools:

{required}

## When to Use Each Tool

- `search_code`: For semantic/code search, conceptual questions, finding examples
- `get_symbol`: For exact symbol/class/method lookup
- `find_callers`: For finding who calls a function/method
- `find_tests_for`: For finding tests that cover a symbol
- `find_related`: For finding related code and dependencies
- `get_file_outline`: For getting file structure before reading full content
- `assemble_context`: For comprehensive context retrieval with callers and tests

## Enforcement Rules

1. Every codebase-related query MUST use at least one Stropha tool
2. Use `get_file_outline` BEFORE reading full files to save context
3. Use `search_code` for conceptual questions, NOT `grep` or manual file reading
4. Use `find_callers` to understand dependencies and impact
5. Use `find_tests_for` to identify relevant tests
6. Chain tools when needed: search → get_file_outline → read specific sections

## Example Workflow

For "How does authentication work?":

1. `search_code(query="authentication", top_k=5)`
2. For each relevant file: `get_file_outline(path="path/to/file.py")`
3. Read specific line ranges of interest
4. `find_callers(symbol="AuthService.login")` to find usage

## Consequences of Non-Compliance

- Responses without proper tool usage will be flagged
- Code analysis may be incomplete or outdated
- You may miss critical dependencies or tests

Remember: Stropha tools provide semantic understanding that text search cannot match. Always prefer them for codebase exploration."""


def get_request_filter_enforcement(enforcement: ToolEnforcementConfig) -> str:
    """Get enforcement instructions for the request filter phase."""
    
    if not enforcement.enabled:
        return ""
    
    return f"""

# MANDATORY STROPHA USAGE FOR REQUEST FILTER

You are the REQUEST FILTER. You MUST use Stropha MCP tools for all codebase exploration.

## Required Actions

1. Use `search_code` for semantic searches (not grep/ripgrep)
2. Use `get_symbol` for exact symbol lookups
3. Use `find_callers` to find usage sites
4. Use `find_tests_for` to find relevant tests
5. Use `assemble_context` for comprehensive feature tracing

## Search Phases

PHASE 1 (SEARCH): Use `search_code` with semantic queries
PHASE 2 (TERM SWEEP): Use `get_symbol` for exact matches
PHASE 3 (PROVENANCE): Use `find_related` for context
PHASE 4 (REQUIRED SLOTS): Use `find_callers` and `find_tests_for`
PHASE 5 (GAPS): Identify missing tool usage
PHASE 6 (VERIFICATION): Confirm all codebase queries used Stropha

## Enforcement

Every search, read, and analysis step MUST be backed by Stropha tool calls.
If you cannot find something with Stropha, mark it as NOT_SEARCHED in PHASE 4.

Your PHASE 6 verification MUST confirm all codebase exploration used Stropha tools."""


def get_contributor_enforcement(enforcement: ToolEnforcementConfig) -> str:
    """Get enforcement instructions for contributor council members."""
    
    if not enforcement.enabled:
        return ""
    
    return f"""

# MANDATORY STROPHA USAGE FOR COUNCIL CONTRIBUTOR

You are a CONTRIBUTOR COUNCIL member. You MUST use Stropha MCP tools for all codebase exploration.

## Required Actions

1. Use `search_code` for semantic understanding of requirements
2. Use `get_symbol` to locate relevant classes/functions
3. Use `find_callers` to understand impact and usage
4. Use `find_tests_for` to identify test coverage
5. Use `find_related` to find related patterns and anti-patterns
6. Use `assemble_context` for end-to-end feature tracing

## Your Five Perspectives

Each perspective MUST use Stropha tools:

### Contrarian
- Use `find_callers` to find failure modes and edge cases
- Use `find_tests_for` to find missing test coverage
- Use `search_code` to find similar failures in codebase

### Software Architect
- Use `get_symbol` to find system boundaries
- Use `find_callers` to understand dependency direction
- Use `find_related` to find architectural patterns

### Clean Coder
- Use `get_symbol` to find code to assess
- Use `find_tests_for` to find test coverage
- Use `find_callers` to understand usage patterns

### Pragmatic Engineer
- Use `assemble_context` to trace feature end-to-end
- Use `find_callers` to find all usage sites
- Use `search_code` to find similar implementations

### Engineering Manager
- Use `find_callers` to find scope of changes
- Use `find_tests_for` to find test requirements
- Use `search_code` to find similar trade-offs

## Enforcement

Every analysis, assessment, and recommendation MUST be backed by Stropha tool calls.
Your responses should reference specific tool results with file:line anchors.

Failure to use Stropha tools will result in incomplete or inaccurate analysis."""


def get_aggregator_enforcement(enforcement: ToolEnforcementConfig) -> str:
    """Get enforcement instructions for the aggregator."""
    
    if not enforcement.enabled:
        return ""
    
    return f"""

# MANDATORY STROPHA USAGE FOR AGGREGATOR

You are the AGGREGATOR. You MUST use Stropha MCP tools to validate and synthesize contributions.

## Required Actions

1. Use `find_callers` to validate contributor claims about usage
2. Use `find_tests_for` to verify test coverage claims
3. Use `search_code` to find conflicting patterns
4. Use `assemble_context` to trace end-to-end flows
5. Use `get_symbol` to verify exact symbol definitions

## Validation Process

For each contributor's analysis:

1. Use `find_callers(symbol=...)` to verify claimed usage sites
2. Use `find_tests_for(symbol=...)` to verify test coverage
3. Use `search_code(query=...)` to find counter-examples
4. Use `assemble_context(task=...)` to verify end-to-end flow
5. Use `get_symbol(symbol=...)` to verify definitions

## Synthesis Requirements

Your final synthesis MUST:
- Reference specific Stropha tool results
- Include file:line anchors from tool outputs
- Cross-validate conflicting claims using Stropha
- Identify gaps that Stropha tools should have filled

## Enforcement

Every claim, validation, and synthesis step MUST reference Stropha tool results.
Responses without Stropha-backed analysis will be incomplete."""


def get_enforcement_summary(enforcement: ToolEnforcementConfig) -> str:
    """Get a summary of tool enforcement for documentation."""
    
    if not enforcement.enabled:
        return "Tool enforcement: DISABLED"
    
    return f"""Tool enforcement: ENABLED ({enforcement.enforcement_mode})
Required tools: {', '.join(enforcement.required_tools)}"""
