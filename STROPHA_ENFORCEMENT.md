# Stropha MCP Tool Enforcement Implementation

## Summary

This implementation enforces the usage of Stropha MCP tools throughout the MoA Gateway for all codebase-related queries.

## Changes Made

### 1. New Module: `src/moa_gateway/tool_enforcement.py`

Created a new module that provides:
- `ToolEnforcementConfig` dataclass for configuration
- `get_stropha_enforcement_prompt()` - General enforcement instructions
- `get_request_filter_enforcement()` - Enforcement for request filter phase
- `get_contributor_enforcement()` - Enforcement for contributor council members
- `get_aggregator_enforcement()` - Enforcement for aggregator
- `get_enforcement_summary()` - Human-readable summary

### 2. Updated `src/moa_gateway/config.py`

- Added `ToolEnforcementConfig` model
- Added `tool_enforcement` field to `ServerConfig`
- Added validation to ensure `required_tools` is populated when enforcement is enabled

### 3. Updated `src/moa_gateway/gateway.py`

- Imported tool enforcement functions
- Added `tool_enforcement` attribute to `Gateway` class
- Modified `_request_filter_request()` to include enforcement instructions
- Modified `_collect_contributions()` to include enforcement instructions in contributor prompts
- Modified `_aggregation_request()` (changed from static to instance method) to include enforcement instructions
- Added enforcement summary to trace logs on initialization

### 4. Updated `moa.yaml`

Added tool enforcement configuration:
```yaml
server:
  tool_enforcement:
    enabled: true
    required_tools:
      - search_code
      - get_symbol
      - find_callers
      - find_tests_for
      - find_related
      - get_file_outline
      - assemble_context
    enforcement_mode: auto
```

## How It Works

### Request Filter Phase
The request filter receives mandatory Stropha usage instructions in its system prompt, requiring it to use Stropha tools for all codebase exploration during its 6-phase analysis.

### Contributor Phase
Each contributor (council member) receives enforcement instructions that require them to:
- Use Stropha tools for semantic understanding
- Use `get_symbol` for exact symbol lookups
- Use `find_callers` to understand impact and usage
- Use `find_tests_for` to identify test coverage
- Reference specific tool results with file:line anchors

### Aggregator Phase
The aggregator receives enforcement instructions that require it to:
- Use Stropha tools to validate contributor claims
- Cross-validate conflicting claims
- Reference specific Stropha tool results
- Identify gaps that Stropha tools should have filled

## Enforcement Modes

- **warn**: Informs users they must use Stropha tools
- **block**: Indicates responses will be rejected without Stropha usage
- **auto**: States Stropha usage is required for all codebase queries

## Testing

All 65 tests pass, including:
- API tests
- Benchmark tests
- CLI tests
- Config validation tests
- Gateway logic tests
- Provider tests
- Tool tests
- Trace tests

## Configuration

To enable/disable tool enforcement, modify `moa.yaml`:

```yaml
server:
  tool_enforcement:
    enabled: false  # or true
    required_tools: [search_code, get_symbol, ...]
    enforcement_mode: warn|block|auto
```

## Benefits

1. **Consistent Codebase Exploration**: All agents use Stropha's semantic search
2. **Better Context**: Stropha provides deeper understanding than text search
3. **Traceability**: Tool results include file:line anchors for verification
4. **Comprehensive Coverage**: Ensures all aspects (callers, tests, related code) are explored
