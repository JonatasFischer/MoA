from __future__ import annotations

import asyncio
import hmac
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from moa_gateway.config import GatewayConfig, load_config
from moa_gateway.domain import StreamEvent
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
from moa_gateway.provider import Provider, UpstreamError


def create_app(
    config: GatewayConfig | None = None,
    providers: dict[str, Provider] | None = None,
) -> FastAPI:
    resolved_config = config or load_config()
    gateway = Gateway(resolved_config, providers)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if resolved_config.server.warmup_on_startup:
            await gateway.warmup()
        yield
        await gateway.close()

    app = FastAPI(title="MoA Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gateway
    app.state.config = resolved_config

    async def authorize(request: Request) -> None:
        expected = resolved_config.server.api_key()
        if not expected:
            return
        authorization = request.headers.get("authorization", "")
        bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        api_key = request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(value, expected) for value in (bearer, api_key)):
            raise HTTPException(status_code=401, detail="invalid gateway credential")

    def public_model(canonical: Any) -> str:
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

    @app.head("/")
    async def probe() -> JSONResponse:
        return JSONResponse({})

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": "moa-gateway", "status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", dependencies=[Depends(authorize)], include_in_schema=False)
    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models() -> dict[str, Any]:
        data: list[dict[str, Any]] = []
        for profile_name, profile in resolved_config.profiles.items():
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
        model = public_model(canonical)
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
            return StreamingResponse(
                chat_stream(events, model),
                media_type="text/event-stream",
                headers={
                    **diagnostic_headers,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            result = await complete_with_disconnect(
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

    @app.post(
        "/v1/messages", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_message(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_anthropic_request(body)
        model = public_model(canonical)
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
            return StreamingResponse(
                anthropic_stream(events, model),
                media_type="text/event-stream",
                headers={
                    **diagnostic_headers,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            result = await complete_with_disconnect(
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

    @app.post(
        "/v1/responses", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_response(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_responses_request(body)
        model = public_model(canonical)
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
            return StreamingResponse(
                responses_stream(events, model),
                media_type="text/event-stream",
                headers={
                    **diagnostic_headers,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            result = await complete_with_disconnect(
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

    return app
