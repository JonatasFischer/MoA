from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any

import jsonschema

from moa_gateway.config import (
    AiStepConfig,
    FlowConfig,
    GateStepConfig,
    GatewayConfig,
    PromptConfig,
    ToolValidatorConfig,
)
from moa_gateway.domain import (
    CanonicalRequest,
    Completion,
    ProviderMetrics,
    StreamEvent,
    Usage,
    content_text,
    merge_tool_call_deltas,
    request_modalities,
)
from moa_gateway.provider import Provider, UpstreamError, validate_request_modalities
from moa_gateway.trace import TraceRecorder


DELEGATED_INVESTIGATION_MARKER = "Mandatory investigation contract: use Stropha"
OPENCODE_MAINTENANCE_PREFIXES = (
    "Create a new anchored summary from the conversation history.",
)
_TEMPLATE_VARIABLE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


@dataclass(frozen=True, slots=True)
class CompiledFlow:
    name: str
    config: FlowConfig
    steps: dict[str, AiStepConfig | GateStepConfig]
    predecessors: dict[str, tuple[str, ...]]
    source_gates: dict[str, GateStepConfig]


@dataclass(slots=True)
class StepOutcome:
    step_id: str
    completion: Completion | None = None
    error: BaseException | None = None
    sources: list["StepOutcome"] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.completion is not None


@dataclass(slots=True)
class GateState:
    step: GateStepConfig
    expected: tuple[str, ...]
    received: dict[str, StepOutcome] = field(default_factory=dict)
    deadline_at: float | None = None


@dataclass(slots=True)
class FlowRun:
    plan: CompiledFlow
    request: CanonicalRequest
    request_id: str
    results: dict[str, StepOutcome] = field(default_factory=dict)
    inputs: dict[str, list[StepOutcome]] = field(default_factory=dict)
    usages: list[Usage] = field(default_factory=list)
    model_requests: dict[str, CanonicalRequest] = field(default_factory=dict)
    result_step_id: str | None = None
    streamed_result: bool = False


def compile_flows(config: GatewayConfig) -> dict[str, CompiledFlow]:
    plans: dict[str, CompiledFlow] = {}
    for name, flow in config.flows.items():
        steps = flow.step_map
        predecessors: dict[str, list[str]] = {step_id: [] for step_id in steps}
        for step in flow.steps:
            for target in step.targets:
                if target.step != "$return":
                    predecessors[target.step].append(step.id)
        source_gates: dict[str, GateStepConfig] = {}
        for gate in (step for step in flow.steps if isinstance(step, GateStepConfig)):
            for source_id in predecessors[gate.id]:
                source_gates[source_id] = gate
        plans[name] = CompiledFlow(
            name=name,
            config=flow,
            steps=steps,
            predecessors={key: tuple(value) for key, value in predecessors.items()},
            source_gates=source_gates,
        )
    return plans


class FlowExecutor:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        trace: TraceRecorder,
    ) -> None:
        self.config = config
        self.providers = providers
        self.trace = trace
        self.plans = compile_flows(config)

    def public_model(self, request: CanonicalRequest) -> str:
        _, flow = self.config.resolve_flow(request.requested_model)
        return request.requested_model or flow.aliases[0]

    async def complete(
        self,
        request: CanonicalRequest,
        request_id: str,
    ) -> Completion:
        flow_name, _ = self.config.resolve_flow(request.requested_model)
        plan = self.plans[flow_name]
        run = FlowRun(plan=plan, request=request, request_id=request_id)
        self.trace.record(
            "request_started",
            request_id,
            flow_id=flow_name,
            requested_model=request.requested_model,
            messages=request.messages,
            tools=request.tools,
            stream=False,
        )
        try:
            result = await self._execute(run)
        except asyncio.CancelledError:
            self.trace.record("request_cancelled", request_id, flow_id=flow_name)
            raise
        except BaseException as exc:
            self.trace.record(
                "request_failed",
                request_id,
                flow_id=flow_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        result, panel_usage = self._finalize_result(run, result)
        self.trace.record(
            "request_completed",
            request_id,
            flow_id=flow_name,
            node_id=run.result_step_id,
            model=result.model,
            content=result.content,
            tool_calls=result.tool_calls,
            usage=self._usage(result.usage),
            usage_total=self._usage(panel_usage),
        )
        return result

    async def stream(
        self,
        request: CanonicalRequest,
        request_id: str,
    ) -> AsyncIterator[StreamEvent]:
        flow_name, _ = self.config.resolve_flow(request.requested_model)
        plan = self.plans[flow_name]
        run = FlowRun(plan=plan, request=request, request_id=request_id)
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        failure: list[BaseException] = []
        self.trace.record(
            "request_started",
            request_id,
            flow_id=flow_name,
            requested_model=request.requested_model,
            messages=request.messages,
            tools=request.tools,
            stream=True,
        )

        async def execute() -> None:
            try:
                result = await self._execute(run, queue)
                result, panel_usage = self._finalize_result(run, result)
                if not run.streamed_result:
                    if result.content:
                        await queue.put(StreamEvent(content=result.content))
                    if result.tool_calls:
                        await queue.put(StreamEvent(tool_calls=result.tool_calls))
                await queue.put(
                    StreamEvent(
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                        metrics=result.metrics,
                        done=True,
                    )
                )
                self.trace.record(
                    "request_completed",
                    request_id,
                    flow_id=flow_name,
                    node_id=run.result_step_id,
                    model=result.model,
                    content=result.content,
                    tool_calls=result.tool_calls,
                    usage=self._usage(result.usage),
                    usage_total=self._usage(panel_usage),
                    stream=True,
                )
            except asyncio.CancelledError:
                self.trace.record("request_cancelled", request_id, flow_id=flow_name)
                raise
            except BaseException as exc:
                failure.append(exc)
                self.trace.record(
                    "request_failed",
                    request_id,
                    flow_id=flow_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    stream=True,
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            await task
            if failure:
                raise failure[0]
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _execute(
        self,
        run: FlowRun,
        stream_queue: asyncio.Queue[StreamEvent | None] | None = None,
    ) -> Completion:
        plan = run.plan
        start_id = self._select_start(plan.config, run.request)
        self.trace.record(
            "flow_started",
            run.request_id,
            flow_id=plan.name,
            node_id=start_id,
            stage=start_id,
        )
        self.trace.record(
            "request_routed",
            run.request_id,
            flow_id=plan.name,
            node_id=start_id,
            route=start_id,
        )
        activations: list[tuple[str, StepOutcome | None]] = [(start_id, None)]
        running: dict[asyncio.Task[StepOutcome], str] = {}
        started: set[str] = set()
        gate_states = {
            step.id: GateState(step, plan.predecessors[step.id])
            for step in plan.config.steps
            if isinstance(step, GateStepConfig)
        }
        semaphores = {
            gate_id: asyncio.Semaphore(state.step.max_concurrency)
            for gate_id, state in gate_states.items()
        }

        try:
            while activations or running:
                while activations:
                    step_id, incoming = activations.pop(0)
                    if step_id == "$return":
                        if incoming is None or not incoming.succeeded:
                            raise UpstreamError(502, "flow returned without a completion")
                        return self._output_completion(run, incoming)
                    step = plan.steps[step_id]
                    if isinstance(step, GateStepConfig):
                        if incoming is None:
                            raise UpstreamError(502, f"gate {step_id} received no source")
                        gate_result = self._notify_gate(
                            run, gate_states[step_id], incoming
                        )
                        if gate_result is not None:
                            run.results[step_id] = gate_result
                            for target_id in self._targets(step, gate_result):
                                activations.append((target_id, gate_result))
                        continue

                    run.inputs.setdefault(step_id, [])
                    if incoming is not None:
                        run.inputs[step_id].append(incoming)
                    if step_id in started:
                        continue
                    started.add(step_id)
                    source_gate = plan.source_gates.get(step_id)
                    task = asyncio.create_task(
                        self._run_ai_with_gate_limit(
                            run,
                            step,
                            run.inputs[step_id],
                            source_gate,
                            gate_states.get(source_gate.id) if source_gate else None,
                            semaphores.get(source_gate.id) if source_gate else None,
                            (
                                stream_queue
                                if stream_queue is not None
                                and self._streamable(step)
                                else None
                            ),
                        )
                    )
                    running[task] = step_id

                if not running:
                    break
                done, _ = await asyncio.wait(
                    running, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    step_id = running.pop(task)
                    outcome = task.result()
                    run.results[step_id] = outcome
                    step = plan.steps[step_id]
                    if outcome.error and not any(
                        target.step in gate_states for target in step.targets
                    ):
                        raise outcome.error
                    targets = self._targets(step, outcome)
                    if not targets and outcome.error:
                        raise outcome.error
                    for target_id in targets:
                        activations.append((target_id, outcome))
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
        raise UpstreamError(502, f"flow {plan.name!r} ended without returning output")

    async def _run_ai_with_gate_limit(
        self,
        run: FlowRun,
        step: AiStepConfig,
        inputs: list[StepOutcome],
        source_gate: GateStepConfig | None,
        gate_state: GateState | None,
        semaphore: asyncio.Semaphore | None,
        stream_queue: asyncio.Queue[StreamEvent | None] | None = None,
    ) -> StepOutcome:
        if source_gate is None or gate_state is None or semaphore is None:
            return await self._run_ai(run, step, inputs, stream_queue)
        loop = asyncio.get_running_loop()
        if gate_state.deadline_at is None and source_gate.deadline_seconds is not None:
            gate_state.deadline_at = loop.time() + source_gate.deadline_seconds
        timeout = (
            max(0.0, gate_state.deadline_at - loop.time())
            if gate_state.deadline_at is not None
            else None
        )
        async def limited() -> StepOutcome:
            async with semaphore:
                return await self._run_ai(run, step, inputs, stream_queue)

        try:
            if timeout is None:
                return await limited()
            return await asyncio.wait_for(limited(), timeout=timeout)
        except TimeoutError as exc:
            self.trace.record(
                "model_failed",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                error_type="TimeoutError",
                error="gate deadline exceeded",
            )
            return StepOutcome(step.id, error=exc)

    async def _run_ai(
        self,
        run: FlowRun,
        step: AiStepConfig,
        inputs: list[StepOutcome],
        stream_queue: asyncio.Queue[StreamEvent | None] | None = None,
    ) -> StepOutcome:
        try:
            request = self._build_request(run, step, inputs)
            emitted_content = False
            if stream_queue is None:
                completion = await self._complete_model(run, step, request, attempt=1)
            else:
                completion, emitted_content = await self._stream_model(
                    run, step, request, stream_queue
                )
            completion = await self._repair_if_needed(
                run, step, inputs, completion
            )
            completion = await self._retry_if_needed(
                run, step, request, completion
            )
            if self._empty(completion) and step.fallback:
                completion = self._fallback(run, step.fallback.gate)
            completion = self._validate_tools(run, step, completion)
            if stream_queue is not None:
                if completion.content and not emitted_content:
                    await stream_queue.put(StreamEvent(content=completion.content))
                if completion.tool_calls:
                    await stream_queue.put(StreamEvent(tool_calls=completion.tool_calls))
                run.streamed_result = True
            return StepOutcome(step.id, completion=completion)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.trace.record(
                "step_failed",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return StepOutcome(step.id, error=exc)

    async def _complete_model(
        self,
        run: FlowRun,
        step: AiStepConfig,
        request: CanonicalRequest,
        *,
        attempt: int,
    ) -> Completion:
        started = time.perf_counter()
        run.model_requests[step.id] = request
        self.trace.record(
            "model_started",
            run.request_id,
            flow_id=run.plan.name,
            node_id=step.id,
            stage=step.id,
            provider=step.provider,
            model=step.model,
            role=step.role,
            family=step.family,
            messages=request.messages,
            tools=request.tools,
            attempt=attempt,
        )
        try:
            validate_request_modalities(
                self.config.providers[step.provider], step.model, request
            )
            completion = await self.providers[step.provider].complete(step.model, request)
        except asyncio.CancelledError:
            self.trace.record(
                "model_cancelled",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                provider=step.provider,
                model=step.model,
                attempt=attempt,
            )
            raise
        except BaseException as exc:
            self.trace.record(
                "model_failed",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                provider=step.provider,
                model=step.model,
                error_type=type(exc).__name__,
                error=str(exc),
                attempt=attempt,
            )
            raise
        run.usages.append(completion.usage)
        self.trace.record(
            "model_completed",
            run.request_id,
            flow_id=run.plan.name,
            node_id=step.id,
            stage=step.id,
            provider=step.provider,
            model=step.model,
            role=step.role,
            family=step.family,
            duration_seconds=round(time.perf_counter() - started, 3),
            content=completion.content,
            tool_calls=completion.tool_calls,
            finish_reason=completion.finish_reason,
            usage=self._usage(completion.usage),
            provider_metrics=self._metrics(completion.metrics),
            attempt=attempt,
        )
        return completion

    async def _stream_model(
        self,
        run: FlowRun,
        step: AiStepConfig,
        request: CanonicalRequest,
        queue: asyncio.Queue[StreamEvent | None],
    ) -> tuple[Completion, bool]:
        started = time.perf_counter()
        run.model_requests[step.id] = request
        self.trace.record(
            "model_started",
            run.request_id,
            flow_id=run.plan.name,
            node_id=step.id,
            stage=step.id,
            provider=step.provider,
            model=step.model,
            role=step.role,
            family=step.family,
            messages=request.messages,
            tools=request.tools,
            attempt=1,
            stream=True,
        )
        content: list[str] = []
        tool_fragments: list[dict[str, Any]] = []
        usage = Usage()
        metrics = ProviderMetrics()
        finish_reason = "stop"
        emitted_content = False
        try:
            validate_request_modalities(
                self.config.providers[step.provider], step.model, request
            )
            async for event in self.providers[step.provider].stream(step.model, request):
                if event.content is not None:
                    content.append(event.content)
                    emitted_content = True
                    await queue.put(StreamEvent(content=event.content))
                if event.tool_calls:
                    tool_fragments.extend(event.tool_calls)
                if event.usage:
                    usage = event.usage
                if event.metrics:
                    metrics = event.metrics
                if event.finish_reason:
                    finish_reason = event.finish_reason
                self.trace.record(
                    "model_delta",
                    run.request_id,
                    flow_id=run.plan.name,
                    node_id=step.id,
                    stage=step.id,
                    provider=step.provider,
                    model=step.model,
                    content=event.content,
                    tool_calls=event.tool_calls,
                    finish_reason=event.finish_reason,
                    done=event.done,
                )
        except asyncio.CancelledError:
            self.trace.record(
                "model_cancelled",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                provider=step.provider,
                model=step.model,
                attempt=1,
                stream=True,
            )
            raise
        completion = Completion(
            content="".join(content),
            model=step.model,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=merge_tool_call_deltas(tool_fragments),
            metrics=metrics,
        )
        run.usages.append(usage)
        self.trace.record(
            "model_completed",
            run.request_id,
            flow_id=run.plan.name,
            node_id=step.id,
            stage=step.id,
            provider=step.provider,
            model=step.model,
            role=step.role,
            family=step.family,
            duration_seconds=round(time.perf_counter() - started, 3),
            content=completion.content,
            tool_calls=completion.tool_calls,
            finish_reason=completion.finish_reason,
            usage=self._usage(completion.usage),
            provider_metrics=self._metrics(completion.metrics),
            attempt=1,
            stream=True,
        )
        return completion, emitted_content

    def _build_request(
        self,
        run: FlowRun,
        step: AiStepConfig,
        inputs: list[StepOutcome],
        *,
        prompt_override: PromptConfig | None = None,
    ) -> CanonicalRequest:
        conversation = self._conversation(run.request.messages, step.conversation)
        messages: list[dict[str, Any]] = []
        prompt = prompt_override or (
            self.config.prompts[step.prompt] if step.prompt else None
        )
        variables = self._template_variables(run, step, inputs, conversation)
        if prompt:
            messages.append(
                {"role": "system", "content": self._render(prompt.system, variables)}
            )
        messages.extend(conversation)
        if prompt and prompt.context:
            messages.append(
                {"role": "user", "content": self._render(prompt.context, variables)}
            )
        tools = self._step_tools(run.request, step)
        if step.max_tokens == "request":
            max_tokens = (
                run.request.max_tokens + step.reasoning_reserve
                if run.request.max_tokens is not None
                else None
            )
        else:
            max_tokens = step.max_tokens
        return CanonicalRequest(
            requested_model=None,
            messages=messages,
            max_tokens=max_tokens,
            temperature=(
                step.temperature
                if step.temperature is not None
                else run.request.temperature
            ),
            stop=None,
            tools=tools,
            tool_choice=self._tool_choice(run.request.tool_choice, tools),
            think=step.think,
            keep_alive=step.keep_alive,
            num_ctx=step.num_ctx,
            response_format=(
                self.config.schemas[step.response_schema]
                if step.response_schema
                else None
            ),
        )

    def _template_variables(
        self,
        run: FlowRun,
        step: AiStepConfig,
        inputs: list[StepOutcome],
        conversation: list[dict[str, Any]],
    ) -> dict[str, str]:
        latest_request = self._latest_user_request(run.request)
        successful = [item for item in inputs if item.succeeded]
        if len(successful) == 1:
            input_text = successful[0].completion.content
        else:
            input_text = json.dumps(
                [
                    {
                        "step": item.step_id,
                        "content": item.completion.content,
                        "model": item.completion.model,
                    }
                    for item in successful
                ],
                ensure_ascii=True,
            )
        variables = {
            "request": latest_request,
            "conversation": json.dumps(
                conversation, ensure_ascii=True, separators=(",", ":")
            ),
            "tools": json.dumps(
                [
                    {
                        "name": tool.get("function", {}).get("name"),
                        "description": tool.get("function", {}).get("description"),
                    }
                    for tool in run.request.tools
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            if run.request.tools
            else "[NOT_IN_CONTEXT]",
            "inputs": input_text,
            "investigation_results": json.dumps(
                [
                    message.get("content", "")
                    for message in run.request.messages
                    if message.get("role") == "tool"
                ],
                ensure_ascii=True,
            ),
            "remaining_investigations": str(
                self._remaining_tool_calls(run.request, step)
            ),
            "role": step.role,
            "family": step.family or step.model,
            **step.prompt_variables,
        }
        for step_id, outcome in run.results.items():
            if outcome.succeeded:
                variables[f"steps.{step_id}"] = outcome.completion.content
        return variables

    @staticmethod
    def _render(template: str, variables: dict[str, str]) -> str:
        return _TEMPLATE_VARIABLE.sub(
            lambda match: variables.get(match.group(1).strip(), match.group(0)),
            template,
        )

    @staticmethod
    def _conversation(
        messages: list[dict[str, Any]], mode: str
    ) -> list[dict[str, Any]]:
        if mode == "none":
            return []
        if mode == "full":
            return [dict(message) for message in messages]
        normalized: list[dict[str, Any]] = []
        remaining = 12000
        for message in reversed(messages):
            if message.get("role") in {"system", "tool"}:
                continue
            clean = dict(message)
            calls = clean.pop("tool_calls", None)
            if calls and not str(clean.get("content") or "").strip():
                continue
            raw_content = clean.get("content") or ""
            if isinstance(raw_content, list):
                content_size = len(
                    json.dumps(raw_content, ensure_ascii=True, separators=(",", ":"))
                )
                if content_size > remaining:
                    continue
                content: Any = [dict(block) for block in raw_content]
            else:
                content = str(raw_content)[:remaining]
                content_size = len(content)
            if not content or remaining <= 0:
                continue
            clean["content"] = content
            remaining -= content_size
            normalized.append(clean)
        return list(reversed(normalized))

    async def _repair_if_needed(
        self,
        run: FlowRun,
        step: AiStepConfig,
        inputs: list[StepOutcome],
        completion: Completion,
    ) -> Completion:
        if not step.response_schema:
            return completion
        if self._schema_valid(completion.content, self.config.schemas[step.response_schema]):
            return completion
        if not step.repair:
            raise UpstreamError(502, f"step {step.id} returned invalid structured output")
        current = completion
        prompt = self.config.prompts[step.repair.prompt]
        for attempt in range(1, step.repair.attempts + 1):
            repair_input = StepOutcome(step.id, completion=current)
            request = self._build_request(
                run, step, [repair_input], prompt_override=prompt
            )
            current = await self._complete_model(
                run, step, request, attempt=attempt + 1
            )
            if self._schema_valid(
                current.content, self.config.schemas[step.response_schema]
            ):
                return current
        raise UpstreamError(502, f"step {step.id} could not repair structured output")

    async def _retry_if_needed(
        self,
        run: FlowRun,
        step: AiStepConfig,
        request: CanonicalRequest,
        completion: Completion,
    ) -> Completion:
        if not step.retry or not self._empty(completion):
            return completion
        current = completion
        current_request = request
        for attempt in range(1, step.retry.attempts + 1):
            current_request = replace(
                current_request,
                max_tokens=(
                    int(current_request.max_tokens * step.retry.max_tokens_multiplier)
                    if current_request.max_tokens is not None
                    else None
                ),
                think=step.retry.think,
            )
            self.trace.record(
                "model_retrying",
                run.request_id,
                flow_id=run.plan.name,
                node_id=step.id,
                stage=step.id,
                reason="empty_completion",
                attempt=attempt + 1,
            )
            current = await self._complete_model(
                run, step, current_request, attempt=attempt + 1
            )
            if not self._empty(current):
                return current
        return current

    def _fallback(self, run: FlowRun, gate_id: str) -> Completion:
        gate = run.results.get(gate_id)
        if gate is None:
            raise UpstreamError(502, f"fallback gate {gate_id!r} has no result")
        usable = [
            item.completion
            for item in gate.sources
            if item.succeeded and item.completion.content.strip()
        ]
        if not usable:
            raise UpstreamError(502, "no non-empty fallback result is available")
        return max(
            usable,
            key=lambda item: (item.finish_reason != "length", len(item.content)),
        )

    def _notify_gate(
        self,
        run: FlowRun,
        state: GateState,
        incoming: StepOutcome,
    ) -> StepOutcome | None:
        state.received[incoming.step_id] = incoming
        successes = [item for item in state.received.values() if item.succeeded]
        failures = len(state.received) - len(successes)
        self.trace.record(
            "gate_progress",
            run.request_id,
            flow_id=run.plan.name,
            node_id=state.step.id,
            stage=state.step.id,
            successes=len(successes),
            failures=failures,
            pending=len(state.expected) - len(state.received),
            min_success=state.step.min_success,
        )
        release = len(state.received) == len(state.expected)
        if state.step.completion == "quorum" and len(successes) >= state.step.min_success:
            release = True
        if not release:
            return None
        if len(successes) < state.step.min_success:
            error = UpstreamError(
                502,
                f"gate {state.step.id} quorum not met: "
                f"{len(successes)}/{state.step.min_success}",
            )
            self.trace.record(
                "gate_failed",
                run.request_id,
                flow_id=run.plan.name,
                node_id=state.step.id,
                stage=state.step.id,
                error=str(error),
            )
            raise error
        ordered = [
            state.received[source]
            for source in state.expected
            if source in state.received and state.received[source].succeeded
        ]
        content = json.dumps(
            [
                {
                    "step": item.step_id,
                    "model": item.completion.model,
                    "content": item.completion.content,
                }
                for item in ordered
            ],
            ensure_ascii=True,
        )
        outcome = StepOutcome(
            state.step.id,
            completion=Completion(content=content, model=f"gate:{state.step.id}"),
            sources=ordered,
        )
        self.trace.record(
            "gate_completed",
            run.request_id,
            flow_id=run.plan.name,
            node_id=state.step.id,
            stage=state.step.id,
            successes=len(ordered),
        )
        return outcome

    def _targets(
        self,
        step: AiStepConfig | GateStepConfig,
        outcome: StepOutcome,
    ) -> list[str]:
        has_calls = bool(outcome.completion and outcome.completion.tool_calls)
        result: list[str] = []
        for target in step.targets:
            if target.when == "always":
                result.append(target.step)
            elif target.when == "has_tool_calls" and has_calls:
                result.append(target.step)
            elif target.when == "no_tool_calls" and not has_calls:
                result.append(target.step)
        return result

    def _step_tools(
        self, request: CanonicalRequest, step: AiStepConfig
    ) -> list[dict[str, Any]]:
        if step.tools.mode == "none":
            return []
        include = set(step.tools.include)
        exclude = set(step.tools.exclude)
        if step.tools.max_calls is not None and self._remaining_tool_calls(request, step) <= 0:
            return []
        return [
            tool
            for tool in request.tools
            if (not include or self._tool_name(tool) in include)
            and self._tool_name(tool) not in exclude
        ]

    def _validate_tools(
        self,
        run: FlowRun,
        step: AiStepConfig,
        completion: Completion,
    ) -> Completion:
        if not completion.tool_calls:
            return completion
        if not step.tools.validator:
            raise UpstreamError(502, f"step {step.id} emitted tools without a validator")
        validator = self.config.tool_validators[step.tools.validator]
        available = {self._tool_name(tool) for tool in self._step_tools(run.request, step)}
        allowed = (
            available
            if validator.allowed_tools == "client"
            else available & set(validator.allowed_tools)
        )
        remaining = self._remaining_tool_calls(run.request, step)
        validated: list[dict[str, Any]] = []
        for call in completion.tool_calls:
            function = dict(call.get("function") or {})
            name = str(function.get("name") or "")
            if name not in allowed:
                raise UpstreamError(502, f"step {step.id} emitted disallowed tool {name!r}")
            if validator.require_call_id and not call.get("id"):
                raise UpstreamError(502, f"tool call {name!r} is missing an id")
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    value = json.loads(arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise UpstreamError(502, f"tool {name!r} arguments are invalid JSON") from exc
            else:
                value = arguments
            if not isinstance(value, dict):
                raise UpstreamError(502, f"tool {name!r} arguments must be an object")
            if transform := validator.transforms.get(name):
                value.update(transform.force_arguments)
                prompt = str(value.get(transform.prompt_field) or "").rstrip()
                if transform.prepend_latest_user_request:
                    original = self._latest_user_request(run.request)
                    if original and original not in prompt:
                        prompt = f"Original user request:\n{original}\n\n{prompt}".rstrip()
                if transform.append_prompt_if_missing and not prompt.endswith(
                    transform.append_prompt_if_missing.strip()
                ):
                    prompt = (
                        prompt + "\n\n" + transform.append_prompt_if_missing.strip()
                    ).strip()
                value[transform.prompt_field] = prompt
            function["arguments"] = json.dumps(
                value, ensure_ascii=True, separators=(",", ":")
            )
            validated.append({**call, "function": function})
            if step.tools.max_calls is not None and len(validated) >= remaining:
                break
        content = "" if validator.mixed_text == "discard" else completion.content
        self.trace.record(
            "tool_calls_validated",
            run.request_id,
            flow_id=run.plan.name,
            node_id=step.id,
            stage=step.id,
            tools=[call["function"]["name"] for call in validated],
        )
        return replace(
            completion,
            content=content,
            tool_calls=validated,
            finish_reason="tool_calls",
        )

    def _output_completion(
        self, run: FlowRun, outcome: StepOutcome
    ) -> Completion:
        run.result_step_id = outcome.step_id
        output = run.plan.config.output
        if (
            outcome.step_id == output.step
            and output.passthrough_input_on_no_tool_calls
            and outcome.completion
            and not outcome.completion.tool_calls
        ):
            inputs = run.inputs.get(outcome.step_id, [])
            usable = [item for item in inputs if item.succeeded]
            if len(usable) != 1:
                raise UpstreamError(
                    502, f"output step {outcome.step_id} needs one passthrough input"
                )
            return usable[0].completion
        return outcome.completion

    @staticmethod
    def _select_start(flow: FlowConfig, request: CanonicalRequest) -> str:
        for start in flow.starts:
            if start.when == "always":
                return start.step
            if start.when == "investigation_result" and FlowExecutor._has_tool_result(request):
                return start.step
            if start.when == "tool_continuation" and FlowExecutor._has_tool_continuation(request):
                return start.step
            if start.when == "opencode_maintenance" and FlowExecutor._is_maintenance(request):
                return start.step
            if start.when == "delegated_investigation" and FlowExecutor._is_delegated(request):
                return start.step
            if start.when == "simple_request" and FlowExecutor._is_simple(flow, request):
                return start.step
        raise UpstreamError(502, "flow has no matching start step")

    @staticmethod
    def _is_simple(flow: FlowConfig, request: CanonicalRequest) -> bool:
        routing = flow.routing
        if routing is None or request_modalities(request) != {"text"}:
            return False
        if routing.require_no_tools and request.tools:
            return False
        if len(request.messages) > routing.max_messages:
            return False
        conversation_chars = sum(
            len(content_text(message.get("content"))) for message in request.messages
        )
        if conversation_chars > routing.max_conversation_chars:
            return False
        latest = FlowExecutor._latest_user_request(request)
        return bool(latest) and len(latest) <= routing.max_latest_user_chars

    @staticmethod
    def _has_tool_result(request: CanonicalRequest) -> bool:
        if not request.messages or request.messages[-1].get("role") != "tool":
            return False
        call_id = request.messages[-1].get("tool_call_id")
        for message in reversed(request.messages[:-1]):
            for call in message.get("tool_calls") or []:
                if call.get("id") == call_id:
                    return call.get("function", {}).get("name") == "task"
        return False

    @staticmethod
    def _has_tool_continuation(request: CanonicalRequest) -> bool:
        if not request.messages:
            return False
        if request.messages[-1].get("role") == "tool":
            return True
        latest = request.messages[-1]
        return bool(
            request.tools
            and latest.get("role") == "assistant"
            and latest.get("tool_calls")
        )

    @staticmethod
    def _is_maintenance(request: CanonicalRequest) -> bool:
        if request.tools or not request.messages:
            return False
        content = str(request.messages[-1].get("content") or "")
        return content.startswith(OPENCODE_MAINTENANCE_PREFIXES)

    @staticmethod
    def _is_delegated(request: CanonicalRequest) -> bool:
        return any(
            DELEGATED_INVESTIGATION_MARKER in str(message.get("content") or "")
            for message in request.messages
            if message.get("role") == "user"
        )

    @staticmethod
    def _latest_user_request(request: CanonicalRequest) -> str:
        return next(
            (
                content_text(message.get("content")).strip()
                for message in reversed(request.messages)
                if message.get("role") == "user"
                and content_text(message.get("content")).strip()
            ),
            "",
        )

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str:
        return str(tool.get("function", {}).get("name") or "")

    @staticmethod
    def _tool_choice(choice: Any, tools: list[dict[str, Any]]) -> Any:
        if not tools:
            return None
        if not isinstance(choice, dict):
            return choice
        selected = str(choice.get("function", {}).get("name") or "")
        available = {FlowExecutor._tool_name(tool) for tool in tools}
        return choice if selected in available else "auto"

    @staticmethod
    def _remaining_tool_calls(request: CanonicalRequest, step: AiStepConfig) -> int:
        if step.tools.max_calls is None:
            return 2**31 - 1
        names = set(step.tools.include)
        previous = sum(
            call.get("function", {}).get("name") in names
            for message in request.messages
            for call in (message.get("tool_calls") or [])
        )
        return max(0, step.tools.max_calls - previous)

    @staticmethod
    def _schema_valid(content: str, schema: dict[str, Any]) -> bool:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                return False
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return False
        try:
            jsonschema.validate(value, schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            return False
        return True

    @staticmethod
    def _empty(completion: Completion) -> bool:
        return not completion.content.strip() and not completion.tool_calls

    @staticmethod
    def _sum_usage(*usages: Usage) -> Usage:
        return Usage(
            input_tokens=sum(usage.input_tokens for usage in usages),
            output_tokens=sum(usage.output_tokens for usage in usages),
            cached_input_tokens=sum(usage.cached_input_tokens for usage in usages),
            reasoning_output_tokens=sum(
                usage.reasoning_output_tokens for usage in usages
            ),
        )

    @staticmethod
    def _client_usage(request: CanonicalRequest, completion: Completion) -> Usage:
        canonical_input = json.dumps(
            {
                "messages": request.messages,
                "tools": request.tools,
                "tool_choice": request.tool_choice,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        input_tokens = completion.usage.input_tokens
        if input_tokens == 0:
            input_tokens = (len(canonical_input) + 3) // 4
        output_tokens = completion.usage.output_tokens
        if output_tokens == 0:
            output = completion.content + json.dumps(
                completion.tool_calls, ensure_ascii=True, separators=(",", ":")
            )
            output_tokens = (len(output) + 3) // 4
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=completion.usage.cached_input_tokens,
            reasoning_output_tokens=completion.usage.reasoning_output_tokens,
        )

    @staticmethod
    def _usage(usage: Usage) -> dict[str, int]:
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_output_tokens": usage.reasoning_output_tokens,
        }

    def _finalize_result(
        self, run: FlowRun, result: Completion
    ) -> tuple[Completion, Usage]:
        panel_usage = self._sum_usage(*run.usages)
        model_request = next(
            (
                run.model_requests[step_id]
                for step_id, outcome in run.results.items()
                if outcome.completion is result and step_id in run.model_requests
            ),
            run.request,
        )
        result = replace(
            result,
            usage=self._client_usage(model_request, result),
            panel_usage=panel_usage if len(run.usages) > 1 else result.panel_usage,
        )
        return result, panel_usage

    def _streamable(self, step: AiStepConfig) -> bool:
        if len(step.targets) != 1:
            return False
        target = step.targets[0]
        if target.step != "$return" or target.when != "always" or step.response_schema:
            return False
        if step.tools.validator:
            validator = self.config.tool_validators[step.tools.validator]
            if validator.mixed_text == "discard" or validator.transforms:
                return False
        return True

    @staticmethod
    def _metrics(metrics: ProviderMetrics) -> dict[str, float]:
        return {
            "total_duration_seconds": metrics.total_duration_ns / 1_000_000_000,
            "load_duration_seconds": metrics.load_duration_ns / 1_000_000_000,
            "prompt_eval_duration_seconds": metrics.prompt_eval_duration_ns
            / 1_000_000_000,
            "eval_duration_seconds": metrics.eval_duration_ns / 1_000_000_000,
        }

    async def warmup(self, flow_names: list[str]) -> None:
        targets: list[AiStepConfig] = []
        names = flow_names or list(self.plans)
        for name in names:
            _, flow = self.config.resolve_flow(name)
            targets.extend(
                step for step in flow.steps if isinstance(step, AiStepConfig)
            )
        seen: set[tuple[str, str]] = set()
        for step in targets:
            if (step.provider, step.model) in seen:
                continue
            seen.add((step.provider, step.model))
            warmup_id = f"warmup:{step.provider}:{step.model}"
            try:
                await self.providers[step.provider].complete(
                    step.model,
                    CanonicalRequest(
                        requested_model=None,
                        messages=[{"role": "user", "content": "Reply with OK only."}],
                        max_tokens=8,
                        temperature=0,
                        think=False,
                        keep_alive=step.keep_alive,
                        num_ctx=step.num_ctx,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self.trace.record(
                    "warmup_failed",
                    warmup_id,
                    provider=step.provider,
                    model=step.model,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            else:
                self.trace.record(
                    "warmup_completed",
                    warmup_id,
                    provider=step.provider,
                    model=step.model,
                )
