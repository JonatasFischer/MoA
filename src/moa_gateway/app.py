from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from moa_gateway.config import (
    ExperimentConfig,
    GatewayConfig,
    ProviderConfig,
    load_config,
)
from moa_gateway.domain import CanonicalRequest, StreamEvent
from moa_gateway.gateway import Gateway
from moa_gateway.protocols import (
    anthropic_message,
    anthropic_stream,
    chat_completion,
    chat_stream,
    parse_anthropic_request,
    parse_chat_request,
    parse_responses_request,
    responses_object,
    responses_stream,
)
from moa_gateway.provider import Provider, UpstreamError, discover_models
from moa_gateway.runtime import GatewayRuntime


class SimulationRequest(BaseModel):
    profile: str
    input: str = Field(min_length=1)
    max_tokens: int = Field(default=1024, ge=1, le=65536)


def create_app(
    config: GatewayConfig | None = None,
    providers: dict[str, Provider] | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    resolved_path = (
        Path(config_path or os.getenv("MOA_CONFIG", "moa.yaml"))
        if config is None
        else (Path(config_path) if config_path is not None else None)
    )
    resolved_config = config or load_config(resolved_path)
    runtime = GatewayRuntime(resolved_config, providers, resolved_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if resolved_config.server.warmup_on_startup:
            await runtime.warmup()
        yield
        await runtime.close()

    app = FastAPI(title="MoA Gateway", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    static_path = Path(__file__).with_name("static")
    app.mount("/assets", StaticFiles(directory=static_path), name="assets")

    async def authorize(request: Request) -> None:
        expected = runtime.config.server.api_key()
        if not expected:
            return
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(value, expected) for value in (bearer, api_key)):
            raise HTTPException(status_code=401, detail="invalid gateway credential")

    def public_model(gateway: Gateway, canonical: Any) -> str:
        try:
            return gateway.public_model(canonical)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"unknown gateway model: {exc.args[0]}"
            ) from exc

    def upstream_error(
        exc: UpstreamError,
        protocol: str,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        if protocol == "anthropic":
            body = {
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            }
        else:
            body = {
                "error": {
                    "type": "api_error",
                    "code": "upstream_error",
                    "message": str(exc),
                }
            }
        return JSONResponse(body, status_code=502, headers=headers)

    def request_context(request: Request) -> tuple[str, str | None, dict[str, str]]:
        request_id = uuid.uuid4().hex
        parent = request.headers.get("x-moa-parent-request-id")
        if parent and (
            len(parent) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", parent)
        ):
            raise HTTPException(status_code=400, detail="invalid parent request id")
        headers = {"X-MoA-Request-ID": request_id}
        if parent:
            headers["X-MoA-Parent-Request-ID"] = parent
        return request_id, parent, headers

    async def prefetch_stream(
        events: AsyncIterator[StreamEvent],
        protocol: str,
        headers: dict[str, str],
    ) -> AsyncIterator[StreamEvent] | JSONResponse:
        try:
            first = await anext(events)
        except UpstreamError as exc:
            return upstream_error(exc, protocol, headers)
        except StopAsyncIteration:
            return upstream_error(
                UpstreamError(502, "upstream stream ended without a response"),
                protocol,
                headers,
            )

        async def replay() -> AsyncIterator[StreamEvent]:
            yield first
            async for event in events:
                yield event

        return replay()

    async def complete_with_disconnect(
        gateway: Gateway,
        http_request: Request,
        canonical: Any,
        request_id: str,
        parent_request_id: str | None,
    ) -> Any:
        task = asyncio.create_task(
            gateway.complete(
                canonical,
                request_id=request_id,
                parent_request_id=parent_request_id,
            )
        )
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if not task.done() and await http_request.is_disconnected():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise HTTPException(status_code=499, detail="client disconnected")
            return await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def release_stream(
        events: AsyncIterator[StreamEvent], lease: Any
    ) -> AsyncIterator[StreamEvent]:
        try:
            async for event in events:
                yield event
        finally:
            await lease.__aexit__(None, None, None)

    def control_payload(config: GatewayConfig) -> dict[str, Any]:
        return {
            "config": ExperimentConfig.from_gateway(config).model_dump(
                by_alias=True, exclude_none=True
            ),
            "generation": runtime.generation,
            "persisted": runtime.persisted,
        }

    @app.head("/")
    async def probe() -> JSONResponse:
        return JSONResponse({})

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(static_path / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    async def get_experiment_config() -> dict[str, Any]:
        return control_payload(runtime.config)

    @app.put("/api/config")
    async def update_experiment_config(
        experiment: ExperimentConfig,
    ) -> dict[str, Any]:
        try:
            updated = await runtime.reconfigure(experiment)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=json.loads(exc.json(include_url=False)),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"could not persist configuration: {exc}",
            ) from exc
        return control_payload(updated)

    @app.get("/api/providers/{provider_name}/models")
    async def provider_models(provider_name: str) -> dict[str, list[str]]:
        provider = runtime.config.providers.get(provider_name)
        if provider is None:
            raise HTTPException(status_code=404, detail="unknown provider")
        try:
            models = await discover_models(provider)
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"models": models}

    @app.post("/api/providers/models")
    async def draft_provider_models(
        provider: ProviderConfig,
    ) -> dict[str, list[str]]:
        try:
            models = await discover_models(provider)
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"models": models}

    @app.post("/api/simulations", response_model=None)
    async def run_simulation(
        simulation: SimulationRequest,
    ) -> StreamingResponse:
        try:
            runtime.config.resolve_profile(simulation.profile)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown profile") from exc

        request_id = uuid.uuid4().hex

        async def events() -> AsyncIterator[str]:
            async with runtime.lease() as gateway:
                queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

                def receive(event: dict[str, Any]) -> None:
                    queue.put_nowait(event)

                enforcement = gateway.config.server.tool_enforcement
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": (
                                "Run a private investigation for this simulation."
                            ),
                            "parameters": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                    }
                    for name in enforcement.investigation_tools
                    if enforcement.enabled
                ]
                canonical = CanonicalRequest(
                    requested_model=simulation.profile,
                    messages=[{"role": "user", "content": simulation.input}],
                    max_tokens=simulation.max_tokens,
                    tools=tools,
                    tool_choice="auto" if tools else None,
                )
                gateway.trace.subscribe(request_id, receive)
                task = asyncio.create_task(
                    gateway.complete(canonical, request_id=request_id)
                )
                started = {
                    "event": "simulation_started",
                    "request_id": request_id,
                    "profile": simulation.profile,
                    "generation": runtime.generation,
                    "available_tools": [
                        tool["function"]["name"] for tool in tools
                    ],
                }
                yield f"data: {json.dumps(started, ensure_ascii=True)}\n\n"
                try:
                    while not task.done() or not queue.empty():
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=0.1)
                        except TimeoutError:
                            continue
                        yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
                    try:
                        result = task.result()
                    except asyncio.CancelledError:
                        raise
                    except BaseException as exc:
                        failed = {
                            "event": "simulation_failed",
                            "request_id": request_id,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                        yield f"data: {json.dumps(failed, ensure_ascii=True)}\n\n"
                    else:
                        completed = {
                            "event": "simulation_completed",
                            "request_id": request_id,
                            "model": result.model,
                            "content": result.content,
                            "tool_calls": result.tool_calls,
                            "finish_reason": result.finish_reason,
                            "usage": {
                                "input_tokens": result.usage.input_tokens,
                                "output_tokens": result.usage.output_tokens,
                            },
                        }
                        yield f"data: {json.dumps(completed, ensure_ascii=True)}\n\n"
                finally:
                    gateway.trace.unsubscribe(request_id, receive)
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-MoA-Request-ID": request_id,
            },
        )

    @app.get("/models", dependencies=[Depends(authorize)], include_in_schema=False)
    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        async with runtime.lease() as gateway:
            for profile_name, profile in gateway.config.profiles.items():
                for alias in profile.aliases:
                    data.append(
                        {
                            "id": alias,
                            "object": "model",
                            "created": 0,
                            "owned_by": "moa-gateway",
                            "display_name": f"MoA Gateway: {profile_name}",
                        }
                    )
        return {"object": "list", "data": data, "has_more": False}

    @app.post(
        "/chat/completions",
        dependencies=[Depends(authorize)],
        response_model=None,
        include_in_schema=False,
    )
    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(authorize)],
        response_model=None,
    )
    async def create_chat(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_chat_request(body)
        lease = runtime.lease()
        gateway = await lease.__aenter__()
        release_lease = True
        try:
            model = public_model(gateway, canonical)
            request_id, parent_id, diagnostic_headers = request_context(request)
            if body.get("stream"):
                events = await prefetch_stream(
                    gateway.stream(
                        canonical,
                        request_id=request_id,
                        parent_request_id=parent_id,
                    ),
                    "openai",
                    diagnostic_headers,
                )
                if isinstance(events, JSONResponse):
                    return events
                release_lease = False
                return StreamingResponse(
                    chat_stream(release_stream(events, lease), model),
                    media_type="text/event-stream",
                    headers={
                        **diagnostic_headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            try:
                result = await complete_with_disconnect(
                    gateway,
                    request,
                    canonical,
                    request_id,
                    parent_id,
                )
            except UpstreamError as exc:
                return upstream_error(exc, "openai", diagnostic_headers)
            return JSONResponse(
                chat_completion(result, model), headers=diagnostic_headers
            )
        finally:
            if release_lease:
                await lease.__aexit__(None, None, None)

    @app.post(
        "/v1/messages", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_message(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_anthropic_request(body)
        lease = runtime.lease()
        gateway = await lease.__aenter__()
        release_lease = True
        try:
            model = public_model(gateway, canonical)
            request_id, parent_id, diagnostic_headers = request_context(request)
            if body.get("stream"):
                events = await prefetch_stream(
                    gateway.stream(
                        canonical,
                        request_id=request_id,
                        parent_request_id=parent_id,
                    ),
                    "anthropic",
                    diagnostic_headers,
                )
                if isinstance(events, JSONResponse):
                    return events
                release_lease = False
                return StreamingResponse(
                    anthropic_stream(release_stream(events, lease), model),
                    media_type="text/event-stream",
                    headers={
                        **diagnostic_headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            try:
                result = await complete_with_disconnect(
                    gateway,
                    request,
                    canonical,
                    request_id,
                    parent_id,
                )
            except UpstreamError as exc:
                return upstream_error(exc, "anthropic", diagnostic_headers)
            return JSONResponse(
                anthropic_message(result, model), headers=diagnostic_headers
            )
        finally:
            if release_lease:
                await lease.__aexit__(None, None, None)

    @app.post(
        "/v1/responses", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_response(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_responses_request(body)
        lease = runtime.lease()
        gateway = await lease.__aenter__()
        release_lease = True
        try:
            model = public_model(gateway, canonical)
            request_id, parent_id, diagnostic_headers = request_context(request)
            if body.get("stream"):
                events = await prefetch_stream(
                    gateway.stream(
                        canonical,
                        request_id=request_id,
                        parent_request_id=parent_id,
                    ),
                    "openai",
                    diagnostic_headers,
                )
                if isinstance(events, JSONResponse):
                    return events
                release_lease = False
                return StreamingResponse(
                    responses_stream(release_stream(events, lease), model),
                    media_type="text/event-stream",
                    headers={
                        **diagnostic_headers,
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            try:
                result = await complete_with_disconnect(
                    gateway,
                    request,
                    canonical,
                    request_id,
                    parent_id,
                )
            except UpstreamError as exc:
                return upstream_error(exc, "openai", diagnostic_headers)
            return JSONResponse(
                responses_object(result, model), headers=diagnostic_headers
            )
        finally:
            if release_lease:
                await lease.__aexit__(None, None, None)

    return app
