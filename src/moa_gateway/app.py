from __future__ import annotations

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from moa_gateway.config import GatewayConfig, load_config
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

    def upstream_error(exc: UpstreamError, protocol: str) -> JSONResponse:
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
        return JSONResponse(body, status_code=502)

    @app.head("/")
    async def probe() -> JSONResponse:
        return JSONResponse({})

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": "moa-gateway", "status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
        "/v1/chat/completions",
        dependencies=[Depends(authorize)],
        response_model=None,
    )
    async def create_chat(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_chat_request(body)
        model = public_model(canonical)
        if body.get("stream"):
            return StreamingResponse(
                chat_stream(gateway.stream(canonical), model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await gateway.complete(canonical)
        except UpstreamError as exc:
            return upstream_error(exc, "openai")
        return JSONResponse(chat_completion(result, model))

    @app.post(
        "/v1/messages", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_message(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_anthropic_request(body)
        model = public_model(canonical)
        if body.get("stream"):
            return StreamingResponse(
                anthropic_stream(gateway.stream(canonical), model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await gateway.complete(canonical)
        except UpstreamError as exc:
            return upstream_error(exc, "anthropic")
        return JSONResponse(anthropic_message(result, model))

    @app.post(
        "/v1/responses", dependencies=[Depends(authorize)], response_model=None
    )
    async def create_response(request: Request) -> JSONResponse | StreamingResponse:
        body = await request.json()
        canonical = parse_responses_request(body)
        model = public_model(canonical)
        if body.get("stream"):
            return StreamingResponse(
                responses_stream(gateway.stream(canonical), model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = await gateway.complete(canonical)
        except UpstreamError as exc:
            return upstream_error(exc, "openai")
        return JSONResponse(responses_object(result, model))

    return app
