from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from moa_gateway.config import ProviderConfig
from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent, Usage


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"upstream returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class Provider(Protocol):
    async def complete(self, model: str, request: CanonicalRequest) -> Completion: ...

    def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]: ...

    async def close(self) -> None: ...


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig) -> None:
        headers: dict[str, str] = {}
        if config.api_key_env and (api_key := os.getenv(config.api_key_env)):
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    @staticmethod
    def _payload(
        model: str, request: CanonicalRequest, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop is not None:
            payload["stop"] = request.stop
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        response = await self.client.post(
            "chat/completions", json=self._payload(model, request, stream=False)
        )
        if response.is_error:
            raise UpstreamError(response.status_code, response.text)
        data = response.json()
        choice = data["choices"][0]
        return Completion(
            content=choice.get("message", {}).get("content") or "",
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason") or "stop",
            usage=Usage.from_openai(data.get("usage")),
        )

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        usage = Usage()
        finish_reason = "stop"
        saw_done = False
        async with self.client.stream(
            "POST",
            "chat/completions",
            json=self._payload(model, request, stream=True),
        ) as response:
            if response.is_error:
                body = (await response.aread()).decode(errors="replace")
                raise UpstreamError(response.status_code, body)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    saw_done = True
                    break
                if not raw:
                    continue
                data = json.loads(raw)
                if data.get("usage"):
                    usage = Usage.from_openai(data["usage"])
                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    yield StreamEvent(content=content)

        yield StreamEvent(
            finish_reason=finish_reason, usage=usage, done=True
        )
        if not saw_done:
            return

    async def close(self) -> None:
        await self.client.aclose()
