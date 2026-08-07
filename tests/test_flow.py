from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from moa_gateway.app import create_app
from moa_gateway.config import GatewayConfig, load_config
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent, Usage
from moa_gateway.gateway import Gateway
from moa_gateway.provider import UpstreamError


TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "task",
        "description": "Run a private investigation.",
        "parameters": {"type": "object"},
    },
}
SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "skill",
        "description": "Load a specialized skill listed in the system prompt.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}
SKILL_CATALOG = """<available_skills>
<skill><name>karpathy-guidelines</name><description>Use when writing, reviewing, or refactoring code.</description></skill>
<skill><name>conventional-commits</name><description>Use when creating or reviewing commits.</description></skill>
</available_skills>"""


class FlowProvider:
    def __init__(self, *, investigate: bool = False) -> None:
        self.investigate = investigate
        self.requests: list[tuple[str, CanonicalRequest]] = []

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        self.requests.append((model, request))
        system = str(request.messages[0].get("content") or "")
        if "request assessment agent" in system:
            content = "INVESTIGATION_NEEDED: NO"
        elif "SOLUTION COUNCIL" in system:
            content = self._council_response("solution")
        elif "ARCHITECTURE REINFORCEMENT COUNCIL" in system:
            content = self._council_response("architecture")
        elif "PROJECT PATTERNS COUNCIL" in system:
            content = self._council_response("project_patterns")
        elif "final blocking reinforcement layer" in system:
            if self.investigate:
                return Completion(
                    content="private checker text",
                    model=model,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_task",
                            "type": "function",
                            "function": {
                                "name": "task",
                                "arguments": json.dumps(
                                    {
                                        "description": "Investigate gap",
                                        "prompt": "Resolve the unanswered question.",
                                    }
                                ),
                            },
                        }
                    ],
                )
            content = '{"approved": true}'
        elif "CONTRIBUTOR COUNCIL" in system:
            content = json.dumps(
                {
                    "contrarian": "challenge",
                    "software_architect": "boundaries",
                    "clean_coder": "clarity",
                    "pragmatic_engineer": "verify",
                    "engineering_manager": "scope",
                }
            )
        elif "Continue the original task" in system:
            content = "integrated answer"
        elif "implementing Engineer" in system:
            content = "aggregate answer"
        else:
            content = "direct answer"
        return Completion(
            content=content,
            model=model,
            usage=Usage(input_tokens=2, output_tokens=1),
        )

    @staticmethod
    def _council_response(council: str) -> str:
        return json.dumps(
            {
                "council": council,
                "decision": "accept",
                "summary": f"{council} assessment",
                "findings": [
                    {
                        "principle": "smallest-correct-change",
                        "applicability": "applicable",
                        "status": "preserved",
                        "severity": "none",
                        "evidence": "request evidence",
                        "required_change": "",
                    }
                ],
                "required_changes": [],
                "open_questions": [],
            }
        )

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        completion = await self.complete(model, request)
        yield StreamEvent(content=completion.content)
        yield StreamEvent(
            done=True,
            finish_reason=completion.finish_reason,
            usage=completion.usage,
        )

    async def close(self) -> None:
        return None


def flow_gateway(provider: FlowProvider) -> Gateway:
    config = load_config("moa.yaml")
    return Gateway(config, {"lms": provider, "ollama": provider, "deepseek": provider})


@pytest.mark.asyncio
async def test_configured_flow_returns_aggregate_without_checker_rewrite() -> None:
    provider = FlowProvider()
    gateway = flow_gateway(provider)

    result = await gateway.complete(
        CanonicalRequest("moa-code", [{"role": "user", "content": "change it"}])
    )

    assert result.content == "aggregate answer"
    assert len(provider.requests) == 6
    checker_request = provider.requests[-1][1]
    assert checker_request.tools == []
    assert "aggregate answer" in checker_request.messages[-1]["content"]


@pytest.mark.asyncio
async def test_investigation_checker_emits_grounded_task_only() -> None:
    provider = FlowProvider(investigate=True)
    gateway = flow_gateway(provider)

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "change the scheduler"}],
            tools=[TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    arguments = json.loads(result.tool_calls[0]["function"]["arguments"])
    assert arguments["subagent_type"] == "explore"
    assert "Original user request:\nchange the scheduler" in arguments["prompt"]
    assert "Mandatory investigation contract: use Stropha" in arguments["prompt"]
    aggregate_request = provider.requests[-2][1]
    assert aggregate_request.tools == []
    assert provider.requests[-1][1].tools == [TASK_TOOL]


@pytest.mark.asyncio
async def test_forced_investigation_choice_is_only_forwarded_to_checker() -> None:
    provider = FlowProvider(investigate=True)
    gateway = flow_gateway(provider)
    forced = {"type": "function", "function": {"name": "task"}}

    await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "inspect it"}],
            tools=[TASK_TOOL],
            tool_choice=forced,
        )
    )

    assert provider.requests[-2][1].tool_choice is None
    assert provider.requests[-1][1].tool_choice == forced


@pytest.mark.asyncio
async def test_investigation_result_integrates_then_checks_again() -> None:
    provider = FlowProvider()
    gateway = flow_gateway(provider)
    task_call = {
        "id": "call_task",
        "type": "function",
        "function": {"name": "task", "arguments": "{}"},
    }

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [
                {"role": "user", "content": "change it"},
                {"role": "assistant", "content": "", "tool_calls": [task_call]},
                {
                    "role": "tool",
                    "tool_call_id": "call_task",
                    "content": "investigation evidence",
                },
            ],
            tools=[TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.content == "integrated answer"
    assert len(provider.requests) == 2
    assert "Continue the original task" in provider.requests[0][1].messages[0]["content"]
    assert "final blocking reinforcement layer" in provider.requests[1][1].messages[0]["content"]


@pytest.mark.asyncio
async def test_regular_tool_result_uses_continuation_path() -> None:
    provider = FlowProvider()
    gateway = flow_gateway(provider)
    read_tool = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object"},
        },
    }
    read_call = {
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [
                {"role": "user", "content": "read it"},
                {"role": "assistant", "content": "", "tool_calls": [read_call]},
                {
                    "role": "tool",
                    "tool_call_id": "call_read",
                    "content": "file contents",
                },
            ],
            tools=[read_tool, TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.content == "integrated answer"
    assert len(provider.requests) == 2
    assert [tool["function"]["name"] for tool in provider.requests[0][1].tools] == [
        "read"
    ]


@pytest.mark.asyncio
async def test_new_user_turn_after_completed_tool_loop_runs_initial_flow() -> None:
    provider = FlowProvider()
    gateway = flow_gateway(provider)
    read_call = {
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [
                {"role": "user", "content": "read it"},
                {"role": "assistant", "content": "", "tool_calls": [read_call]},
                {"role": "tool", "tool_call_id": "call_read", "content": "data"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "now do something else"},
            ],
            tools=[TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.content == "aggregate answer"
    assert len(provider.requests) == 6


@pytest.mark.asyncio
async def test_aggregate_client_action_is_validated_and_reinforced() -> None:
    read_tool = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object"},
        },
    }

    class ActionProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            system = str(request.messages[0].get("content") or "")
            if "implementing Engineer" in system:
                self.requests.append((model, request))
                return Completion(
                    content="",
                    model=model,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                )
            return await super().complete(model, request)

    provider = ActionProvider()
    gateway = flow_gateway(provider)

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "read the readme"}],
            tools=[read_tool, TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0]["function"]["name"] == "read"
    reinforcement = provider.requests[-1][1]
    assert "final blocking reinforcement layer" in str(
        reinforcement.messages[0]["content"]
    )
    proposed = json.loads(
        reinforcement.messages[-1]["content"]
        .split("Proposed completion, including tool calls:\n", 1)[1]
        .split("\n\nAvailable skills:", 1)[0]
    )
    assert proposed[0]["tool_calls"][0]["id"] == "call_read"


@pytest.mark.asyncio
async def test_reinforcement_blocks_action_until_required_skill_is_loaded() -> None:
    class SkillProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            system = str(request.messages[0].get("content") or "")
            if "final blocking reinforcement layer" in system:
                self.requests.append((model, request))
                context = str(request.messages[-1].get("content") or "")
                if "Loaded skill guidance:\n[]" in context:
                    return Completion(
                        content="provisional text that must be discarded",
                        model=model,
                        finish_reason="tool_calls",
                        tool_calls=[
                            {
                                "id": "call_skill",
                                "type": "function",
                                "function": {
                                    "name": "skill",
                                    "arguments": json.dumps(
                                        {"name": "karpathy-guidelines"}
                                    ),
                                },
                            }
                        ],
                    )
            return await super().complete(model, request)

    provider = SkillProvider()
    gateway = flow_gateway(provider)
    initial = CanonicalRequest(
        "moa-code",
        [
            {"role": "system", "content": SKILL_CATALOG},
            {"role": "user", "content": "change the implementation"},
        ],
        tools=[SKILL_TOOL, TASK_TOOL],
        tool_choice="auto",
    )

    blocked = await gateway.complete(initial)

    assert blocked.content == ""
    assert blocked.tool_calls[0]["function"]["name"] == "skill"
    assert json.loads(blocked.tool_calls[0]["function"]["arguments"]) == {
        "name": "karpathy-guidelines"
    }

    loaded = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            [
                *initial.messages,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": blocked.tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_skill",
                    "content": "# Skill guidance\nMake the smallest correct change.",
                },
            ],
            tools=[SKILL_TOOL, TASK_TOOL],
            tool_choice="auto",
        )
    )

    second_run = provider.requests[6:]
    assert "request assessment agent" in second_run[0][1].messages[0]["content"]
    assert loaded.content == "aggregate answer"
    aggregate_context = second_run[-2][1].messages[-1]["content"]
    assert "Make the smallest correct change" in aggregate_context


@pytest.mark.asyncio
async def test_reinforcement_rejects_unknown_skill() -> None:
    class UnknownSkillProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            system = str(request.messages[0].get("content") or "")
            if "final blocking reinforcement layer" in system:
                self.requests.append((model, request))
                return Completion(
                    content="",
                    model=model,
                    finish_reason="tool_calls",
                    tool_calls=[
                        {
                            "id": "call_skill",
                            "type": "function",
                            "function": {
                                "name": "skill",
                                "arguments": '{"name":"missing-skill"}',
                            },
                        }
                    ],
                )
            return await super().complete(model, request)

    gateway = flow_gateway(UnknownSkillProvider())

    with pytest.raises(UpstreamError, match="unknown skill 'missing-skill'"):
        await gateway.complete(
            CanonicalRequest(
                "moa-code",
                [
                    {"role": "system", "content": SKILL_CATALOG},
                    {"role": "user", "content": "change it"},
                ],
                tools=[SKILL_TOOL],
            )
        )


def test_flow_config_rejects_cycles() -> None:
    config = load_config("moa.yaml").model_dump(by_alias=True)
    direct = config["flows"]["direct"]
    direct["steps"][0]["targets"] = [{"step": "answer"}]

    with pytest.raises(ValueError, match="cycle"):
        GatewayConfig.model_validate(config)


@pytest.mark.asyncio
async def test_v2_config_api_reconfigures_models_and_next_request() -> None:
    provider = FlowProvider()
    config = load_config("moa.yaml")
    app = create_app(
        config,
        {"lms": provider, "ollama": provider, "deepseek": provider},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        models = await client.get("/v1/models")
        payload = (await client.get("/api/config")).json()["config"]
        payload["flows"]["direct"]["steps"][0]["model"] = "replacement-model"
        updated = await client.put("/api/config", json=payload)
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "direct-code",
                "messages": [{"role": "user", "content": "answer"}],
            },
        )

    assert models.status_code == 200
    assert "moa-code" in {item["id"] for item in models.json()["data"]}
    assert updated.status_code == 200
    assert updated.json()["generation"] == 2
    assert response.status_code == 200
    assert provider.requests[-1][0] == "replacement-model"


@pytest.mark.asyncio
async def test_v2_config_api_persists_version_flows_and_warmups(tmp_path) -> None:
    provider = FlowProvider()
    config = load_config("moa.yaml")
    path = tmp_path / "flows.yaml"
    app = create_app(
        config,
        {"lms": provider, "ollama": provider, "deepseek": provider},
        config_path=path,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = (await client.get("/api/config")).json()["config"]
        payload["flows"]["renamed-code"] = payload["flows"].pop("code")
        payload["default_flow"] = "renamed-code"
        payload["warmup_flows"] = ["renamed-code"]
        response = await client.put("/api/config", json=payload)

    persisted = load_config(path)
    assert response.status_code == 200
    assert persisted.version == 2
    assert persisted.default_flow == "renamed-code"
    assert persisted.server.warmup_flows == ["renamed-code"]


def test_flow_config_rejects_conditional_gate_sources_and_stranded_steps() -> None:
    config = load_config("moa.yaml").model_dump(by_alias=True)
    contributor = next(
        step
        for step in config["flows"]["code"]["steps"]
        if step["id"] == "solution-council"
    )
    contributor["targets"][0]["when"] = "no_tool_calls"
    with pytest.raises(ValueError, match="cannot have conditional sources"):
        GatewayConfig.model_validate(config)

    config = load_config("moa.yaml").model_dump(by_alias=True)
    config["flows"]["direct"]["steps"][0]["targets"] = []
    with pytest.raises(ValueError, match="cannot reach \\$return"):
        GatewayConfig.model_validate(config)


def test_flow_config_rejects_ambiguous_fan_in_and_conditional_dead_ends() -> None:
    config = load_config("moa.yaml").model_dump(by_alias=True)
    checker = next(
        step
        for step in config["flows"]["code"]["steps"]
        if step["id"] == "action-skill-reinforcement"
    )
    checker["activation"] = "single"
    with pytest.raises(ValueError, match="multiple sources"):
        GatewayConfig.model_validate(config)

    config = load_config("moa.yaml").model_dump(by_alias=True)
    direct = config["flows"]["direct"]["steps"][0]
    direct["targets"] = [{"step": "$return", "when": "has_tool_calls"}]
    with pytest.raises(ValueError, match="incomplete conditional targets"):
        GatewayConfig.model_validate(config)

    config = load_config("moa.yaml").model_dump(by_alias=True)
    aggregate = next(
        step
        for step in config["flows"]["code"]["steps"]
        if step["id"] == "aggregate"
    )
    aggregate["fallback"]["gate"] = "request-filter"
    with pytest.raises(ValueError, match="fallback must reference a gate"):
        GatewayConfig.model_validate(config)


@pytest.mark.asyncio
async def test_gate_deadline_includes_time_waiting_for_concurrency() -> None:
    class SlowProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            system = str(request.messages[0].get("content") or "")
            if "COUNCIL" in system:
                await asyncio.sleep(0.2)
            return await super().complete(model, request)

    config = load_config("moa.yaml")
    raw = config.model_dump(by_alias=True)
    gate = next(
        step
        for step in raw["flows"]["code"]["steps"]
        if step["id"] == "contributions"
    )
    gate["max_concurrency"] = 1
    gate["deadline_seconds"] = 0.05
    provider = SlowProvider()
    gateway = Gateway(
        GatewayConfig.model_validate(raw),
        {"lms": provider, "ollama": provider, "deepseek": provider},
    )

    started = time.perf_counter()
    with pytest.raises(UpstreamError, match="quorum not met"):
        await gateway.complete(
            CanonicalRequest("moa-code", [{"role": "user", "content": "run"}])
        )

    assert time.perf_counter() - started < 0.15


@pytest.mark.asyncio
async def test_flow_stream_emits_terminal_error_after_progress() -> None:
    class FailingProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            raise UpstreamError(500, "failed")

    provider = FailingProvider()
    gateway = flow_gateway(provider)

    events = [
        event
        async for event in gateway.stream(
            CanonicalRequest("moa-code", [{"role": "user", "content": "run"}])
        )
    ]

    assert events[0].progress == "executing configured flow"
    assert events[-1].done is True
    assert "failed" in events[-1].error


@pytest.mark.asyncio
async def test_flow_stream_converts_transport_exception_to_terminal_error() -> None:
    class FailingProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            raise RuntimeError("connection lost")

    gateway = flow_gateway(FailingProvider())

    events = [
        event
        async for event in gateway.stream(
            CanonicalRequest("moa-code", [{"role": "user", "content": "run"}])
        )
    ]

    assert events[-1].done is True
    assert "connection lost" in events[-1].error


@pytest.mark.asyncio
async def test_warmup_failure_does_not_prevent_startup() -> None:
    class FailingProvider(FlowProvider):
        async def complete(self, model: str, request: CanonicalRequest) -> Completion:
            raise UpstreamError(500, "offline")

    gateway = flow_gateway(FailingProvider())

    await gateway.warmup()


@pytest.mark.asyncio
async def test_reinforcement_limit_removes_private_tools() -> None:
    provider = FlowProvider()
    gateway = flow_gateway(provider)
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "task", "arguments": "{}"},
        }
        for index in range(4)
    ]
    messages: list[dict[str, object]] = [{"role": "user", "content": "change it"}]
    for call in calls:
        messages.extend(
            [
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "evidence",
                },
            ]
        )

    result = await gateway.complete(
        CanonicalRequest(
            "moa-code",
            messages,
            tools=[TASK_TOOL],
            tool_choice="auto",
        )
    )

    assert result.content == "integrated answer"
    assert provider.requests[-1][1].tools == []


@pytest.mark.asyncio
async def test_terminal_flow_step_streams_provider_deltas() -> None:
    class StreamingProvider:
        def __init__(self) -> None:
            self.streamed = False

        async def complete(
            self, model: str, request: CanonicalRequest
        ) -> Completion:
            raise AssertionError("terminal step should use provider streaming")

        async def stream(
            self, model: str, request: CanonicalRequest
        ) -> AsyncIterator[StreamEvent]:
            self.streamed = True
            yield StreamEvent(content="first ")
            yield StreamEvent(content="second")
            yield StreamEvent(
                finish_reason="stop",
                usage=Usage(input_tokens=7, output_tokens=2),
                done=True,
            )

        async def close(self) -> None:
            return None

    provider = StreamingProvider()
    gateway = Gateway(
        load_config("moa.yaml"),
        {"lms": provider, "ollama": provider, "deepseek": provider},
    )

    events = [
        event
        async for event in gateway.stream(
            CanonicalRequest("direct-code", [{"role": "user", "content": "run"}])
        )
    ]

    assert provider.streamed is True
    assert [event.content for event in events if event.content] == ["first ", "second"]
    assert events[-1].done is True
    assert events[-1].usage == Usage(input_tokens=7, output_tokens=2)


@pytest.mark.asyncio
async def test_flow_usage_preserves_visible_model_and_tracks_panel() -> None:
    provider = FlowProvider()
    result = await flow_gateway(provider).complete(
        CanonicalRequest(
            "moa-code",
            [{"role": "user", "content": "Review this substantial implementation"}],
        )
    )

    assert result.usage == Usage(input_tokens=2, output_tokens=1)
    assert result.panel_usage == Usage(input_tokens=12, output_tokens=6)


@pytest.mark.asyncio
async def test_simple_request_start_routes_deterministically() -> None:
    raw = load_config("moa.yaml").model_dump(by_alias=True)
    direct = raw["flows"]["direct"]
    answer = direct["steps"][0]
    simple = json.loads(json.dumps(answer))
    simple["id"] = "simple-answer"
    simple["model"] = "simple-model"
    direct["steps"].insert(0, simple)
    direct["starts"] = [
        {"step": "simple-answer", "when": "simple_request"},
        {"step": "answer", "when": "always"},
    ]
    direct["routing"] = {
        "max_latest_user_chars": 10,
        "max_conversation_chars": 20,
        "max_messages": 1,
        "require_no_tools": True,
    }
    provider = FlowProvider()
    gateway = Gateway(
        GatewayConfig.model_validate(raw),
        {"lms": provider, "ollama": provider, "deepseek": provider},
    )

    await gateway.complete(
        CanonicalRequest("direct-code", [{"role": "user", "content": "short"}])
    )
    await gateway.complete(
        CanonicalRequest(
            "direct-code",
            [{"role": "user", "content": "this request exceeds the threshold"}],
        )
    )

    assert [model for model, _ in provider.requests] == [
        "simple-model",
        "Qwen/Qwen3-Coder-Next-FP8",
    ]


@pytest.mark.asyncio
async def test_start_priority_overrides_declaration_order() -> None:
    raw = load_config("moa.yaml").model_dump(by_alias=True)
    direct = raw["flows"]["direct"]
    answer = direct["steps"][0]
    simple = json.loads(json.dumps(answer))
    simple["id"] = "simple-answer"
    simple["model"] = "simple-model"
    direct["steps"].insert(0, simple)
    direct["starts"] = [
        {"step": "answer", "when": "always", "priority": 20},
        {"step": "simple-answer", "when": "simple_request", "priority": 10},
    ]
    direct["routing"] = {
        "max_latest_user_chars": 10,
        "max_conversation_chars": 20,
        "max_messages": 1,
        "require_no_tools": True,
    }
    provider = FlowProvider()
    gateway = Gateway(
        GatewayConfig.model_validate(raw),
        {"lms": provider, "ollama": provider, "deepseek": provider},
    )

    await gateway.complete(
        CanonicalRequest("direct-code", [{"role": "user", "content": "short"}])
    )
    await gateway.complete(
        CanonicalRequest(
            "direct-code",
            [{"role": "user", "content": "this request exceeds the threshold"}],
        )
    )

    assert [model for model, _ in provider.requests] == [
        "simple-model",
        "Qwen/Qwen3-Coder-Next-FP8",
    ]
