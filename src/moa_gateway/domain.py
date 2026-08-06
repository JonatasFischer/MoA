from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalRequest:
    requested_model: str | None
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    think: bool | None = None
    keep_alive: str | int | float | None = None
    num_ctx: int | None = None
    response_format: str | dict[str, Any] | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_openai(cls, value: dict[str, Any] | None) -> "Usage":
        value = value or {}
        prompt_details = value.get("prompt_tokens_details") or {}
        completion_details = value.get("completion_tokens_details") or {}
        return cls(
            input_tokens=int(value.get("prompt_tokens", 0) or 0),
            output_tokens=int(value.get("completion_tokens", 0) or 0),
            cached_input_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
            reasoning_output_tokens=int(
                completion_details.get("reasoning_tokens", 0) or 0
            ),
        )


@dataclass(slots=True)
class ProviderMetrics:
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_duration_ns: int = 0
    eval_duration_ns: int = 0


@dataclass(slots=True)
class Completion:
    content: str
    model: str
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    panel_usage: Usage | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: ProviderMetrics = field(default_factory=ProviderMetrics)


@dataclass(slots=True)
class StreamEvent:
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    progress: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    metrics: ProviderMetrics | None = None
    done: bool = False


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def request_modalities(request: CanonicalRequest) -> set[str]:
    modalities = {"text"}
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                modalities.add("image")
            elif block.get("type") == "file":
                modalities.add("file")
    return modalities


def merge_tool_call_deltas(
    fragments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for position, fragment in enumerate(fragments):
        index = fragment.get("index")
        if not isinstance(index, int):
            index = position
        call = merged.setdefault(index, {"index": index, "function": {}})
        for key in ("id", "type"):
            if fragment.get(key) is not None:
                call[key] = fragment[key]
        source_function = fragment.get("function") or {}
        function = call["function"]
        if source_function.get("name"):
            function["name"] = source_function["name"]
        if source_function.get("arguments") is not None:
            function["arguments"] = str(function.get("arguments") or "") + str(
                source_function["arguments"]
            )
    return [merged[index] for index in sorted(merged)]
