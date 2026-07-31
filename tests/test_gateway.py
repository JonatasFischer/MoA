from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from moa_gateway.config import GatewayConfig
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent, Usage
from moa_gateway.gateway import Gateway
from moa_gateway.provider import UpstreamError


class PanelProvider:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.requests: list[tuple[str, CanonicalRequest]] = []

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        self.requests.append((model, request))
        if model in self.failing:
            raise UpstreamError(500, f"{model} failed")
        return Completion(content=f"advice from {model}", model=model)

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append((model, request))
        yield StreamEvent(content="final")
        yield StreamEvent(done=True, finish_reason="stop")

    async def close(self) -> None:
        return None


def classic_config(min_quorum: int = 1) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {"api_key_env": None},
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://local.test/v1",
                }
            },
            "profiles": {
                "code": {
                    "aliases": ["moa-code"],
                    "strategy": "classic",
                    "proposers": [
                        {"provider": "local", "model": "one", "role": "implementer"},
                        {"provider": "local", "model": "two", "role": "reviewer"},
                    ],
                    "aggregator": {"provider": "local", "model": "final"},
                    "min_quorum": min_quorum,
                    "proposer_max_tokens": 123,
                    "reference_token_budget": 20,
                }
            },
            "default_profile": "code",
        }
    )


def council_config(
    trace_log_path: str | None = None,
    *,
    min_quorum: int = 3,
    deadline_seconds: float | None = None,
    contributor_format: str = "text",
) -> GatewayConfig:
    return GatewayConfig.model_validate(
        {
            "server": {
                "api_key_env": None,
                "trace_log_path": trace_log_path,
            },
            "providers": {
                "local": {
                    "type": "openai-compatible",
                    "base_url": "http://local.test/v1",
                }
            },
            "profiles": {
                "code": {
                    "aliases": ["moa-code"],
                    "strategy": "council",
                    "contributors": [
                        {
                            "provider": "local",
                            "model": "qwen2.5-coder:7b",
                            "family": "qwen",
                        },
                        {
                            "provider": "local",
                            "model": "gemma4:latest",
                            "family": "gemma",
                        },
                        {
                            "provider": "local",
                            "model": "deepseek-coder-v2:16b",
                            "family": "deepseek",
                        },
                    ],
                    "aggregator": {
                        "provider": "local",
                        "model": "qwen3.6:27b",
                        "family": "qwen",
                        "think": False,
                    },
                    "tool_dispatch": {
                        "provider": "local",
                        "model": "qwen3-coder:30b-128k",
                        "family": "qwen",
                        "think": False,
                    },
                    "min_quorum": min_quorum,
                    "contributor_deadline_seconds": deadline_seconds,
                    "contributor_format": contributor_format,
                    "contributor_max_tokens": 1536,
                    "reference_token_budget": 1,
                    "reasoning_reserve": {"qwen": 4096},
                }
            },
            "default_profile": "code",
        }
    )


@pytest.mark.asyncio
async def test_classic_profile_collects_advice_then_aggregates() -> None:
    provider = PanelProvider()
    gateway = Gateway(classic_config(), {"local": provider})
    request = CanonicalRequest(
        requested_model="moa-code",
        messages=[{"role": "user", "content": "solve this"}],
        max_tokens=500,
    )

    result = await gateway.complete(request)

    assert result.model == "final"
    assert [model for model, _ in provider.requests] == ["one", "two", "final"]
    assert provider.requests[0][1].max_tokens == 123
    aggregate = provider.requests[-1][1]
    assert aggregate.max_tokens == 500
    assert "advice from one" in aggregate.messages[-1]["content"]
    assert "advice from two" in aggregate.messages[-1]["content"]
    assert "untrusted" in aggregate.messages[0]["content"]


@pytest.mark.asyncio
async def test_classic_profile_tolerates_failure_when_quorum_is_met() -> None:
    provider = PanelProvider({"two"})
    gateway = Gateway(classic_config(min_quorum=1), {"local": provider})
    request = CanonicalRequest(None, [{"role": "user", "content": "solve"}])

    await gateway.complete(request)

    assert provider.requests[-1][0] == "final"
    assert "advice from one" in provider.requests[-1][1].messages[-1]["content"]


@pytest.mark.asyncio
async def test_classic_profile_rejects_failed_quorum() -> None:
    provider = PanelProvider({"two"})
    gateway = Gateway(classic_config(min_quorum=2), {"local": provider})
    request = CanonicalRequest(None, [{"role": "user", "content": "solve"}])

    with pytest.raises(UpstreamError, match="quorum not met"):
        await gateway.complete(request)


@pytest.mark.asyncio
async def test_classic_streams_only_the_final_model() -> None:
    provider = PanelProvider()
    gateway = Gateway(classic_config(), {"local": provider})
    request = CanonicalRequest(None, [{"role": "user", "content": "solve"}])

    events = [event async for event in gateway.stream(request)]

    assert [event.content for event in events if event.content] == ["final"]
    assert events[0].progress == "collecting contributor quorum"
    assert [model for model, _ in provider.requests] == ["one", "two", "final"]


@pytest.mark.asyncio
async def test_proposers_exclude_agent_prompt_and_tool_history() -> None:
    provider = PanelProvider()
    gateway = Gateway(classic_config(), {"local": provider})
    request = CanonicalRequest(
        None,
        [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
    )

    await gateway.complete(request)

    proposer_request = provider.requests[0][1]
    assert proposer_request.tools == []
    assert all(message["role"] != "tool" for message in proposer_request.messages)
    assert [message["role"] for message in proposer_request.messages] == [
        "system",
        "user",
    ]
    assert proposer_request.messages[1]["content"] == "read"
    aggregate_request = provider.requests[-1][1]
    assert aggregate_request.tools == request.tools
    assert aggregate_request.messages[-2]["role"] == "tool"


@pytest.mark.asyncio
async def test_tool_result_turn_routes_directly_without_council() -> None:
    provider = PanelProvider()
    gateway = Gateway(council_config(), {"local": provider})
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "parameters": {"type": "object"}},
        }
    ]
    request = CanonicalRequest(
        "moa-code",
        [
            {"role": "user", "content": "inspect this"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ],
        tools=tools,
        tool_choice="auto",
    )

    result = await gateway.complete(request)

    assert result.content == "advice from qwen3-coder:30b-128k"
    assert [model for model, _ in provider.requests] == ["qwen3-coder:30b-128k"]
    assert provider.requests[0][1].tools == tools
    assert provider.requests[0][1].think is False


@pytest.mark.asyncio
async def test_each_contributor_runs_full_council_before_qwen_aggregation() -> None:
    provider = PanelProvider()
    gateway = Gateway(council_config(), {"local": provider})

    await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "solve"}],
            max_tokens=512,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                }
            ],
            tool_choice="auto",
        )
    )

    contributor_requests = provider.requests[:3]
    aggregator_request = provider.requests[-1]

    assert [model for model, _ in contributor_requests] == [
        "qwen2.5-coder:7b",
        "gemma4:latest",
        "deepseek-coder-v2:16b",
    ]
    assert aggregator_request[0] == "qwen3.6:27b"
    fields = {
        "contrarian",
        "first_principles_thinker",
        "maintainer",
        "outsider",
        "executor",
    }
    for _, contribution in contributor_requests:
        prompt = contribution.messages[0]["content"]
        assert all(field in prompt for field in fields)
        assert "3-5 substantive" in prompt
        assert contribution.max_tokens == 1536
        assert contribution.tools == []

    aggregate = aggregator_request[1]
    references = aggregate.messages[-1]["content"]
    assert "advice from qwen2.5-coder:7b" in references
    assert "advice from gemma4:latest" in references
    assert "advice from deepseek-coder-v2:16b" in references
    assert "Compare matching perspectives" in references
    assert "You are the Tech Lead" in aggregate.messages[0]["content"]
    assert "unresolved trade-offs" in aggregate.messages[0]["content"]
    assert aggregate.max_tokens == 4608
    assert aggregate.think is False
    assert aggregate.tools[0]["function"]["name"] == "read"
    assert aggregate.tool_choice == "auto"


@pytest.mark.asyncio
async def test_trace_records_complete_model_outputs(tmp_path) -> None:
    trace_path = tmp_path / "moa-trace.jsonl"
    provider = PanelProvider()
    gateway = Gateway(council_config(str(trace_path)), {"local": provider})

    await gateway.complete(
        CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    request_ids = {record["request_id"] for record in records}
    completed = [record for record in records if record["event"] == "model_completed"]

    assert len(request_ids) == 1
    assert [record["stage"] for record in completed] == [
        "contributor",
        "contributor",
        "contributor",
        "aggregator",
    ]
    assert [record["model"] for record in completed] == [
        "qwen2.5-coder:7b",
        "gemma4:latest",
        "deepseek-coder-v2:16b",
        "qwen3.6:27b",
    ]
    assert completed[0]["content"] == "advice from qwen2.5-coder:7b"
    assert records[-1]["event"] == "request_completed"


@pytest.mark.asyncio
async def test_trace_records_parent_request_id(tmp_path) -> None:
    trace_path = tmp_path / "parent-trace.jsonl"
    gateway = Gateway(
        council_config(str(trace_path)), {"local": PanelProvider()}
    )

    await gateway.complete(
        CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}]),
        request_id="child",
        parent_request_id="parent",
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert {record["request_id"] for record in records} == {"child"}
    assert all(record["parent_request_id"] == "parent" for record in records)


@pytest.mark.asyncio
async def test_trace_records_streaming_deltas(tmp_path) -> None:
    trace_path = tmp_path / "moa-stream-trace.jsonl"
    provider = PanelProvider()
    gateway = Gateway(council_config(str(trace_path)), {"local": provider})

    events = [
        event
        async for event in gateway.stream(
            CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
        )
    ]

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    deltas = [record for record in records if record["event"] == "model_delta"]
    aggregator = [
        record
        for record in records
        if record["event"] == "model_completed"
        and record["stage"] == "aggregator"
    ]

    assert [event.content for event in events if event.content] == ["final"]
    assert events[0].progress == "collecting contributor quorum"
    assert deltas[0]["content"] == "final"
    assert aggregator[0]["content"] == "final"


@pytest.mark.asyncio
async def test_empty_aggregation_retries_with_double_budget() -> None:
    class RecoveringProvider(PanelProvider):
        aggregate_attempts = 0

        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            if model != "qwen3.6:27b":
                return Completion(content=f"advice from {model}", model=model)
            self.aggregate_attempts += 1
            if self.aggregate_attempts == 1:
                return Completion(
                    content="",
                    model=model,
                    finish_reason="length",
                    usage=Usage(input_tokens=100, output_tokens=4608),
                )
            return Completion(
                content="recovered answer",
                model=model,
                usage=Usage(input_tokens=100, output_tokens=2),
            )

    provider = RecoveringProvider()
    gateway = Gateway(council_config(), {"local": provider})

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "solve"}],
            max_tokens=512,
        )
    )

    aggregate_requests = [
        request for model, request in provider.requests if model == "qwen3.6:27b"
    ]
    assert result.content == "recovered answer"
    assert [request.max_tokens for request in aggregate_requests] == [4608, 9216]
    assert all(request.think is False for request in aggregate_requests)
    assert result.panel_usage == Usage(input_tokens=200, output_tokens=4610)


@pytest.mark.asyncio
async def test_empty_aggregation_falls_back_to_best_contributor() -> None:
    class FallbackProvider(PanelProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            if model == "qwen3.6:27b":
                return Completion(content="", model=model, finish_reason="length")
            contents = {
                "qwen2.5-coder:7b": "short answer",
                "gemma4:latest": "the strongest complete contributor answer",
                "deepseek-coder-v2:16b": "a much longer but truncated contributor answer",
            }
            return Completion(
                content=contents[model],
                model=model,
                finish_reason=(
                    "length" if model == "deepseek-coder-v2:16b" else "stop"
                ),
            )

    provider = FallbackProvider()
    gateway = Gateway(council_config(), {"local": provider})

    result = await gateway.complete(
        CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
    )

    assert result.content == "the strongest complete contributor answer"
    assert result.model == "gemma4:latest"
    assert sum(model == "qwen3.6:27b" for model, _ in provider.requests) == 2


@pytest.mark.asyncio
async def test_empty_aggregation_without_fallback_is_an_error() -> None:
    class EmptyProvider(PanelProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            return Completion(content="", model=model, finish_reason="length")

    gateway = Gateway(council_config(), {"local": EmptyProvider()})

    with pytest.raises(UpstreamError, match="no contributor fallback"):
        await gateway.complete(
            CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
        )


@pytest.mark.asyncio
async def test_empty_streaming_aggregation_retries_before_emitting() -> None:
    class StreamingRecoveryProvider(PanelProvider):
        aggregate_attempts = 0

        async def stream(
            self, model: str, request: CanonicalRequest
        ) -> AsyncIterator[StreamEvent]:
            self.requests.append((model, request))
            self.aggregate_attempts += 1
            if self.aggregate_attempts == 1:
                yield StreamEvent(
                    finish_reason="length", usage=Usage(output_tokens=4608), done=True
                )
                return
            yield StreamEvent(content="recovered stream")
            yield StreamEvent(finish_reason="stop", usage=Usage(output_tokens=2), done=True)

    provider = StreamingRecoveryProvider()
    gateway = Gateway(council_config(), {"local": provider})

    events = [
        event
        async for event in gateway.stream(
            CanonicalRequest(
                "moa-code",
                [{"role": "user", "content": "solve"}],
                max_tokens=512,
            )
        )
    ]

    assert [event.content for event in events if event.content] == ["recovered stream"]
    aggregate_requests = [
        request for model, request in provider.requests if model == "qwen3.6:27b"
    ]
    assert [request.max_tokens for request in aggregate_requests] == [4608, 9216]


@pytest.mark.asyncio
async def test_contributor_quorum_cancels_and_marks_absent_straggler(tmp_path) -> None:
    class QuorumProvider(PanelProvider):
        cancelled: set[str] = set()

        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            if model == "qwen3.6:27b":
                return Completion(content="final", model=model)
            delay = {
                "qwen2.5-coder:7b": 0.001,
                "gemma4:latest": 0.002,
                "deepseek-coder-v2:16b": 10,
            }[model]
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled.add(model)
                raise
            return Completion(content=f"advice from {model}", model=model)

    trace_path = tmp_path / "quorum-trace.jsonl"
    provider = QuorumProvider()
    gateway = Gateway(
        council_config(
            str(trace_path), min_quorum=2, deadline_seconds=1
        ),
        {"local": provider},
    )

    result = await asyncio.wait_for(
        gateway.complete(
            CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
        ),
        timeout=1,
    )

    assert result.content == "final"
    assert provider.cancelled == {"deepseek-coder-v2:16b"}
    aggregate = provider.requests[-1][1]
    evidence = json.loads(aggregate.messages[-1]["content"].split("\n", 1)[1])
    assert evidence["absent_models"] == ["deepseek-coder-v2:16b"]
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    completed = [record for record in records if record["event"] == "stage_completed"]
    assert completed[0]["successes"] == 2
    assert completed[0]["cancelled_models"] == ["deepseek-coder-v2:16b"]


@pytest.mark.asyncio
async def test_structured_council_is_validated_before_aggregation() -> None:
    class StructuredProvider(PanelProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            if model == "qwen3.6:27b":
                return Completion(content="final", model=model)
            content = json.dumps(
                {
                    "contrarian": "risk",
                    "first_principles_thinker": "essentials",
                    "maintainer": "maintenance",
                    "outsider": "analogy",
                    "executor": "action",
                }
            )
            return Completion(content=content, model=model)

    provider = StructuredProvider()
    gateway = Gateway(
        council_config(contributor_format="json-schema"), {"local": provider}
    )

    result = await gateway.complete(
        CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
    )

    assert result.content == "final"
    for _, request in provider.requests[:3]:
        assert request.response_format["required"] == [
            "contrarian",
            "first_principles_thinker",
            "maintainer",
            "outsider",
            "executor",
        ]
    evidence = json.loads(provider.requests[-1][1].messages[-1]["content"].split("\n", 1)[1])
    assert json.loads(evidence["candidates"][0]["content"])["executor"] == "action"


@pytest.mark.asyncio
async def test_invalid_structured_council_does_not_count_toward_quorum() -> None:
    class InvalidProvider(PanelProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            return Completion(content='{"contrarian":"only one"}', model=model)

    gateway = Gateway(
        council_config(contributor_format="json-schema"),
        {"local": InvalidProvider()},
    )

    with pytest.raises(UpstreamError, match="quorum not met"):
        await gateway.complete(
            CanonicalRequest("moa-code", [{"role": "user", "content": "solve"}])
        )


@pytest.mark.asyncio
async def test_stream_and_non_stream_report_same_client_usage(tmp_path) -> None:
    class UsageProvider(PanelProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            self.requests.append((model, request))
            return Completion(
                content=("final answer" if model == "qwen3.6:27b" else "advice"),
                model=model,
                usage=(
                    Usage(input_tokens=20, output_tokens=2)
                    if model == "qwen3.6:27b"
                    else Usage(input_tokens=10, output_tokens=1)
                ),
            )

        async def stream(
            self, model: str, request: CanonicalRequest
        ) -> AsyncIterator[StreamEvent]:
            self.requests.append((model, request))
            yield StreamEvent(content="final answer")
            yield StreamEvent(
                finish_reason="stop",
                usage=Usage(input_tokens=20, output_tokens=2),
                done=True,
            )

    request = CanonicalRequest(
        "moa-code", [{"role": "user", "content": "solve this"}]
    )
    complete_trace = tmp_path / "complete.jsonl"
    stream_trace = tmp_path / "stream.jsonl"
    complete_gateway = Gateway(
        council_config(str(complete_trace)), {"local": UsageProvider()}
    )
    stream_gateway = Gateway(
        council_config(str(stream_trace)), {"local": UsageProvider()}
    )

    completion = await complete_gateway.complete(request)
    events = [event async for event in stream_gateway.stream(request)]
    final_event = next(event for event in events if event.done)

    assert completion.usage == final_event.usage
    assert completion.panel_usage == Usage(input_tokens=50, output_tokens=5)
    for path in (complete_trace, stream_trace):
        completed = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if json.loads(line)["event"] == "request_completed"
        ][0]
        assert completed["usage"] == {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": 2,
            "total_tokens": completion.usage.total_tokens,
        }
        assert completed["usage_by_stage"]["contributor"]["input_tokens"] == 30
        assert completed["usage_total"]["total_tokens"] == 55


@pytest.mark.asyncio
async def test_warmup_loads_each_configured_model_once() -> None:
    provider = PanelProvider()
    gateway = Gateway(council_config(min_quorum=2), {"local": provider})

    await gateway.warmup()

    assert [model for model, _ in provider.requests] == [
        "qwen2.5-coder:7b",
        "gemma4:latest",
        "qwen3.6:27b",
        "qwen3-coder:30b-128k",
    ]
    assert all(request.max_tokens == 8 for _, request in provider.requests)
    assert all(request.think is False for _, request in provider.requests)
