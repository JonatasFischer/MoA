from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalRequest:
    requested_model: str | None
    messages: list[dict[str, str]]
    max_tokens: int | None = None
    temperature: float | None = None
    stop: str | list[str] | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_openai(cls, value: dict[str, Any] | None) -> "Usage":
        value = value or {}
        return cls(
            input_tokens=int(value.get("prompt_tokens", 0) or 0),
            output_tokens=int(value.get("completion_tokens", 0) or 0),
        )


@dataclass(slots=True)
class Completion:
    content: str
    model: str
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class StreamEvent:
    content: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    done: bool = False
