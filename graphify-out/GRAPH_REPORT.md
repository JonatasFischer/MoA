# Graph Report - MoA  (2026-07-29)

## Corpus Check
- 16 files · ~5,783 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 123 nodes · 301 edges · 11 communities (8 shown, 3 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1e3f23fc`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Configurable Mixture-of-Agents Gateway Plan
- protocols.py
- GatewayConfig
- CanonicalRequest
- cli.py
- Usage
- test_api.py
- OpenAICompatibleProvider
- StreamEvent
- __init__.py
- moa-gateway

## God Nodes (most connected - your core abstractions)
1. `CanonicalRequest` - 23 edges
2. `create_app()` - 19 edges
3. `Completion` - 17 edges
4. `StreamEvent` - 16 edges
5. `Usage` - 15 edges
6. `Provider` - 15 edges
7. `OpenAICompatibleProvider` - 15 edges
8. `Gateway` - 14 edges
9. `GatewayConfig` - 13 edges
10. `Configurable Mixture-of-Agents Gateway Plan` - 13 edges

## Surprising Connections (you probably didn't know these)
- `test_bearer_and_api_key_auth()` --calls--> `create_app()`  [EXTRACTED]
  tests/test_api.py → src/moa_gateway/app.py
- `FakeProvider` --uses--> `CanonicalRequest`  [INFERRED]
  tests/conftest.py → src/moa_gateway/domain.py
- `FakeProvider` --uses--> `Usage`  [INFERRED]
  tests/conftest.py → src/moa_gateway/domain.py
- `FakeProvider` --uses--> `Completion`  [INFERRED]
  tests/conftest.py → src/moa_gateway/domain.py
- `FakeProvider` --uses--> `StreamEvent`  [INFERRED]
  tests/conftest.py → src/moa_gateway/domain.py

## Import Cycles
- None detected.

## Communities (11 total, 3 thin omitted)

### Community 0 - "Configurable Mixture-of-Agents Gateway Plan"
Cohesion: 0.08
Nodes (23): 1. Foundation, 2. Protocol-Compatible Direct Mode, 3. Agent Tool Conformance, 4. Classic MoA, 5. Configuration and Diagnostics, 6. Additional Strategies and Evaluation, Acceptance Criteria, API Surfaces (+15 more)

### Community 1 - "protocols.py"
Cohesion: 0.32
Nodes (20): FastAPI, create_app(), anthropic_message(), _anthropic_stop(), anthropic_stream(), chat_completion(), chat_stream(), _id() (+12 more)

### Community 2 - "GatewayConfig"
Cohesion: 0.19
Nodes (7): BaseModel, GatewayConfig, ProfileConfig, ServerConfig, api(), FakeProvider, gateway_config()

### Community 3 - "CanonicalRequest"
Cohesion: 0.26
Nodes (5): Protocol, CanonicalRequest, Completion, Gateway, Provider

### Community 4 - "cli.py"
Cohesion: 0.24
Nodes (7): ArgumentParser, Path, main(), _parser(), _starter_config(), load_config(), test_loads_yaml_and_resolves_alias()

### Community 5 - "Usage"
Cohesion: 0.29
Nodes (5): Exception, Any, Usage, Any, UpstreamError

### Community 6 - "test_api.py"
Cohesion: 0.22
Nodes (3): _event_types(), test_all_streaming_protocols(), test_bearer_and_api_key_auth()

### Community 7 - "OpenAICompatibleProvider"
Cohesion: 0.33
Nodes (3): ProviderConfig, OpenAICompatibleProvider, test_openai_compatible_complete_and_stream()

## Knowledge Gaps
- **20 isolated node(s):** `moa-gateway`, `Goal`, `Research Basis`, `Chosen Product Direction`, `API Surfaces` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CanonicalRequest` connect `CanonicalRequest` to `protocols.py`, `GatewayConfig`, `Usage`, `OpenAICompatibleProvider`, `StreamEvent`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `create_app()` connect `protocols.py` to `GatewayConfig`, `CanonicalRequest`, `cli.py`, `test_api.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `GatewayConfig` connect `GatewayConfig` to `StreamEvent`, `protocols.py`, `CanonicalRequest`, `cli.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `CanonicalRequest` (e.g. with `Gateway` and `OpenAICompatibleProvider`) actually correct?**
  _`CanonicalRequest` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Completion` (e.g. with `Gateway` and `OpenAICompatibleProvider`) actually correct?**
  _`Completion` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `StreamEvent` (e.g. with `Gateway` and `OpenAICompatibleProvider`) actually correct?**
  _`StreamEvent` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Usage` (e.g. with `OpenAICompatibleProvider` and `Provider`) actually correct?**
  _`Usage` has 4 INFERRED edges - model-reasoned connections that need verification._