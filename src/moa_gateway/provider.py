from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx

from moa_gateway.config import ProviderConfig
from moa_gateway.domain import (
    CanonicalRequest,
    Completion,
    ProviderMetrics,
    StreamEvent,
    Usage,
    content_text,
    request_modalities,
)


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"upstream returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class CapabilityError(Exception):
    pass


def validate_request_modalities(
    config: ProviderConfig, model: str, request: CanonicalRequest
) -> None:
    unsupported = request_modalities(request) - config.modalities_for(model)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise CapabilityError(f"model {model!r} does not support input modalities: {names}")


class Provider(Protocol):
    async def complete(self, model: str, request: CanonicalRequest) -> Completion: ...

    def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]: ...

    async def close(self) -> None: ...


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
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
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            response_format = request.response_format
            if isinstance(response_format, str):
                response_format = {"type": response_format}
            elif response_format.get("type") in {"json_object", "text"}:
                pass
            elif not (
                response_format.get("type") == "json_schema"
                and "json_schema" in response_format
            ):
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_response",
                        "strict": True,
                        "schema": response_format,
                    },
                }
            payload["response_format"] = response_format
        if request.think is not None:
            payload["think"] = request.think
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        validate_request_modalities(self.config, model, request)
        response = await self.client.post(
            "chat/completions", json=self._payload(model, request, stream=False)
        )
        if response.is_error:
            raise UpstreamError(response.status_code, response.text)
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        return Completion(
            content=message.get("content") or "",
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason") or "stop",
            usage=Usage.from_openai(data.get("usage")),
            tool_calls=message.get("tool_calls") or [],
        )

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        validate_request_modalities(self.config, model, request)
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
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    yield StreamEvent(content=content)
                if delta.get("tool_calls"):
                    yield StreamEvent(tool_calls=delta["tool_calls"])

        yield StreamEvent(
            finish_reason=finish_reason, usage=usage, done=True
        )
        if not saw_done:
            return

    async def close(self) -> None:
        await self.client.aclose()


class OpenAIProvider(OpenAICompatibleProvider):
    pass


class DeepSeekProvider(OpenAICompatibleProvider):
    pass


class OllamaProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    @staticmethod
    def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        tool_names: dict[str, str] = {}
        for message in messages:
            clean = dict(message)
            content = clean.get("content")
            if isinstance(content, list):
                images: list[str] = []
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "image_url":
                        continue
                    image = block.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else image
                    if not isinstance(url, str) or not url:
                        raise CapabilityError("Ollama image input requires a non-empty URL")
                    if url.startswith("data:"):
                        try:
                            url = url.split(",", 1)[1]
                        except IndexError as exc:
                            raise CapabilityError("invalid image data URL") from exc
                    images.append(url)
                clean["content"] = content_text(content)
                if images:
                    clean["images"] = images
            calls = clean.get("tool_calls") or []
            normalized_calls: list[dict[str, Any]] = []
            for call in calls:
                normalized = dict(call)
                function = dict(normalized.get("function") or {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise UpstreamError(
                            400, "tool call arguments must be valid JSON"
                        ) from exc
                if not isinstance(arguments, dict):
                    raise UpstreamError(400, "tool call arguments must be an object")
                function["arguments"] = arguments
                normalized["function"] = function
                normalized_calls.append(normalized)
                if normalized.get("id") and function.get("name"):
                    tool_names[str(normalized["id"])] = str(function["name"])
            if normalized_calls:
                clean["tool_calls"] = normalized_calls
            if clean.get("role") == "tool":
                call_id = str(clean.get("tool_call_id") or "")
                tool_name = clean.get("name") or tool_names.get(call_id)
                if tool_name:
                    clean["tool_name"] = tool_name
            result.append(clean)
        return result

    @staticmethod
    def _payload(
        model: str, request: CanonicalRequest, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": OllamaProvider._messages(request.messages),
            "stream": stream,
        }
        options: dict[str, Any] = {}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.stop is not None:
            options["stop"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )
        if request.num_ctx is not None:
            options["num_ctx"] = request.num_ctx
        if options:
            payload["options"] = options
        if request.think is not None:
            payload["think"] = request.think
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        if request.response_format is not None:
            payload["format"] = request.response_format
        if request.tools and request.tool_choice != "none":
            if request.tool_choice not in {None, "auto"}:
                raise UpstreamError(400, "native Ollama only supports automatic tools")
            payload["tools"] = request.tools
        return payload

    @staticmethod
    def _metrics(data: dict[str, Any]) -> ProviderMetrics:
        return ProviderMetrics(
            total_duration_ns=int(data.get("total_duration", 0) or 0),
            load_duration_ns=int(data.get("load_duration", 0) or 0),
            prompt_eval_duration_ns=int(data.get("prompt_eval_duration", 0) or 0),
            eval_duration_ns=int(data.get("eval_duration", 0) or 0),
        )

    @staticmethod
    def _tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, call in enumerate(data.get("tool_calls") or []):
            function = dict(call.get("function") or {})
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, separators=(",", ":"))
            result.append(
                {
                    "id": call.get("id") or f"call_ollama_{index}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )
        return result

    async def complete(self, model: str, request: CanonicalRequest) -> Completion:
        validate_request_modalities(self.config, model, request)
        response = await self.client.post(
            "api/chat", json=self._payload(model, request, stream=False)
        )
        if response.is_error:
            raise UpstreamError(response.status_code, response.text)
        data = response.json()
        if data.get("error"):
            raise UpstreamError(502, str(data["error"]))
        message = data.get("message") or {}
        tool_calls = self._tool_calls(message)
        return Completion(
            content=message.get("content") or "",
            model=data.get("model", model),
            finish_reason="tool_calls" if tool_calls else data.get("done_reason", "stop"),
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                output_tokens=int(data.get("eval_count", 0) or 0),
            ),
            tool_calls=tool_calls,
            metrics=self._metrics(data),
        )

    async def stream(
        self, model: str, request: CanonicalRequest
    ) -> AsyncIterator[StreamEvent]:
        validate_request_modalities(self.config, model, request)
        saw_done = False
        async with self.client.stream(
            "POST", "api/chat", json=self._payload(model, request, stream=True)
        ) as response:
            if response.is_error:
                body = (await response.aread()).decode(errors="replace")
                raise UpstreamError(response.status_code, body)
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise UpstreamError(502, "invalid Ollama stream response") from exc
                if data.get("error"):
                    raise UpstreamError(502, str(data["error"]))
                message = data.get("message") or {}
                content = message.get("content")
                if content:
                    yield StreamEvent(content=content)
                tool_calls = self._tool_calls(message)
                if tool_calls:
                    yield StreamEvent(tool_calls=tool_calls)
                if data.get("done"):
                    saw_done = True
                    yield StreamEvent(
                        finish_reason=(
                            "tool_calls"
                            if tool_calls
                            else data.get("done_reason", "stop")
                        ),
                        usage=Usage(
                            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
                            output_tokens=int(data.get("eval_count", 0) or 0),
                        ),
                        metrics=self._metrics(data),
                        done=True,
                    )
        if not saw_done:
            raise UpstreamError(502, "Ollama stream ended before done=true")

    async def close(self) -> None:
        await self.client.aclose()


def create_provider(config: ProviderConfig) -> Provider:
    provider_type = {
        "openai-compatible": OpenAICompatibleProvider,
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider,
    }[config.kind]
    return provider_type(config)


async def discover_models(config: ProviderConfig) -> list[str]:
    headers: dict[str, str] = {}
    if config.api_key_env and (api_key := os.getenv(config.api_key_env)):
        headers["Authorization"] = f"Bearer {api_key}"
    path = "api/tags" if config.kind == "ollama" else "models"
    async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/") + "/",
        headers=headers,
        timeout=min(config.timeout_seconds, 10),
    ) as client:
        response = await client.get(path)
    if response.is_error:
        raise UpstreamError(response.status_code, response.text)
    try:
        payload = response.json()
        entries = payload.get("models" if config.kind == "ollama" else "data", [])
        names = [entry.get("name") or entry.get("id") for entry in entries]
    except (AttributeError, TypeError, ValueError) as exc:
        raise UpstreamError(502, "provider returned an invalid model list") from exc
    return sorted({str(name) for name in names if name})
