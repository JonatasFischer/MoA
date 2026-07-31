from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException

from moa_gateway.domain import CanonicalRequest, Completion, StreamEvent, Usage


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _require_object(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    return body


def _text_content(content: Any, accepted_types: set[str]) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail="message content must be text")

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in accepted_types:
            raise HTTPException(
                status_code=501,
                detail="only text content is implemented in this milestone",
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="text block is missing text")
        parts.append(text)
    return "".join(parts)


def _reject_tools(body: dict[str, Any]) -> None:
    if body.get("tools") or body.get("functions"):
        raise HTTPException(
            status_code=501,
            detail=(
                "client tool calling is not implemented yet; tools are rejected "
                "rather than silently discarded"
            ),
        )


def _openai_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise HTTPException(status_code=400, detail=f"unsupported message role: {role}")
        content = message.get("content", "")
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        normalized: dict[str, Any] = {
            "role": "system" if role == "developer" else role,
            "content": _text_content(content, {"text"}),
        }
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise HTTPException(status_code=400, detail="tool message needs tool_call_id")
            normalized["tool_call_id"] = tool_call_id
            if isinstance(message.get("name"), str):
                normalized["name"] = message["name"]
        if message.get("tool_calls") is not None:
            if role != "assistant" or not isinstance(message["tool_calls"], list):
                raise HTTPException(status_code=400, detail="invalid assistant tool_calls")
            normalized["tool_calls"] = message["tool_calls"]
        result.append(normalized)
    return result


def _openai_tools(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("functions"):
        raise HTTPException(status_code=501, detail="legacy functions are unsupported")
    tools = data.get("tools") or []
    if not isinstance(tools, list):
        raise HTTPException(status_code=400, detail="tools must be a list")
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type") != "function"
            or not isinstance(tool.get("function"), dict)
            or not isinstance(tool["function"].get("name"), str)
        ):
            raise HTTPException(status_code=400, detail="invalid function tool")
    return tools


def parse_chat_request(body: Any) -> CanonicalRequest:
    data = _require_object(body)
    return CanonicalRequest(
        requested_model=data.get("model"),
        messages=_openai_messages(data.get("messages")),
        max_tokens=data.get("max_completion_tokens", data.get("max_tokens")),
        temperature=data.get("temperature"),
        stop=data.get("stop"),
        tools=_openai_tools(data),
        tool_choice=data.get("tool_choice"),
    )


def parse_anthropic_request(body: Any) -> CanonicalRequest:
    data = _require_object(body)
    _reject_tools(data)
    messages: list[dict[str, str]] = []
    system = data.get("system")
    if system:
        messages.append(
            {
                "role": "system",
                "content": _text_content(system, {"text"}),
            }
        )

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    for message in raw_messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"unsupported message role: {role}")
        messages.append(
            {
                "role": role,
                "content": _text_content(message.get("content", ""), {"text"}),
            }
        )

    return CanonicalRequest(
        requested_model=data.get("model"),
        messages=messages,
        max_tokens=data.get("max_tokens"),
        temperature=data.get("temperature"),
        stop=data.get("stop_sequences"),
    )


def parse_responses_request(body: Any) -> CanonicalRequest:
    data = _require_object(body)
    _reject_tools(data)
    if data.get("previous_response_id") or data.get("conversation"):
        raise HTTPException(
            status_code=501,
            detail="server-managed Responses conversations are not implemented yet",
        )

    messages: list[dict[str, str]] = []
    instructions = data.get("instructions")
    if instructions:
        messages.append(
            {
                "role": "system",
                "content": _text_content(instructions, {"input_text", "text"}),
            }
        )

    value = data.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict) or item.get("type", "message") != "message":
                raise HTTPException(
                    status_code=501,
                    detail="only Responses message input items are implemented",
                )
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise HTTPException(
                    status_code=400, detail=f"unsupported input role: {role}"
                )
            messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _text_content(
                        item.get("content", ""),
                        {"input_text", "output_text", "text"},
                    ),
                }
            )
    else:
        raise HTTPException(status_code=400, detail="input must be text or message items")

    return CanonicalRequest(
        requested_model=data.get("model"),
        messages=messages,
        max_tokens=data.get("max_output_tokens"),
        temperature=data.get("temperature"),
    )


def chat_completion(completion: Completion, public_model: str) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": completion.content or (None if completion.tool_calls else ""),
        "refusal": None,
    }
    if completion.tool_calls:
        message["tool_calls"] = completion.tool_calls
    return {
        "id": _id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": completion.finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": completion.usage.input_tokens,
            "completion_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.total_tokens,
        },
    }


def _anthropic_stop(reason: str | None) -> str:
    return {
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }.get(reason or "stop", "end_turn")


def anthropic_message(completion: Completion, public_model: str) -> dict[str, Any]:
    return {
        "id": _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": public_model,
        "content": [{"type": "text", "text": completion.content}],
        "stop_reason": _anthropic_stop(completion.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
        },
    }


def _responses_usage(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens,
    }


def responses_object(
    completion: Completion,
    public_model: str,
    *,
    response_id: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    response_id = response_id or _id("resp")
    message_id = message_id or _id("msg")
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": public_model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": completion.content,
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "usage": _responses_usage(completion.usage),
    }


def _sse(payload: dict[str, Any], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def chat_stream(
    events: AsyncIterator[StreamEvent], public_model: str
) -> AsyncIterator[str]:
    response_id = _id("chatcmpl")
    created = int(time.time())
    yield _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )
    async for item in events:
        if item.progress:
            yield f": moa-progress {item.progress.replace(chr(10), ' ')}\n\n"
            continue
        if item.error:
            yield _sse(
                {
                    "error": {
                        "type": "api_error",
                        "code": "upstream_error",
                        "message": item.error,
                    }
                }
            )
            return
        if item.content is not None:
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item.content},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        if item.tool_calls:
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": item.tool_calls},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        if item.done:
            yield _sse(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": item.finish_reason or "stop",
                        }
                    ],
                }
            )
            if item.usage:
                yield _sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": public_model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": item.usage.input_tokens,
                            "completion_tokens": item.usage.output_tokens,
                            "total_tokens": item.usage.total_tokens,
                        },
                    }
                )
    yield "data: [DONE]\n\n"


async def anthropic_stream(
    events: AsyncIterator[StreamEvent], public_model: str
) -> AsyncIterator[str]:
    message_id = _id("msg")
    yield _sse(
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": public_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        "message_start",
    )
    yield _sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        "content_block_start",
    )
    async for item in events:
        if item.progress:
            yield f": moa-progress {item.progress.replace(chr(10), ' ')}\n\n"
            continue
        if item.error:
            yield _sse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": item.error},
                },
                "error",
            )
            return
        if item.content is not None:
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": item.content},
                },
                "content_block_delta",
            )
        if item.done:
            usage = item.usage or Usage()
            yield _sse(
                {"type": "content_block_stop", "index": 0},
                "content_block_stop",
            )
            yield _sse(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _anthropic_stop(item.finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": usage.output_tokens},
                },
                "message_delta",
            )
            yield _sse({"type": "message_stop"}, "message_stop")


async def responses_stream(
    events: AsyncIterator[StreamEvent], public_model: str
) -> AsyncIterator[str]:
    response_id = _id("resp")
    message_id = _id("msg")
    sequence = 0
    text = ""

    initial = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": public_model,
        "output": [],
        "error": None,
        "incomplete_details": None,
    }
    yield _sse(
        {"type": "response.created", "sequence_number": sequence, "response": initial},
        "response.created",
    )
    sequence += 1
    yield _sse(
        {
            "type": "response.in_progress",
            "sequence_number": sequence,
            "response": initial,
        },
        "response.in_progress",
    )
    sequence += 1
    output_item = {
        "id": message_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield _sse(
        {
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": 0,
            "item": output_item,
        },
        "response.output_item.added",
    )
    sequence += 1
    part = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
    yield _sse(
        {
            "type": "response.content_part.added",
            "sequence_number": sequence,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": part,
        },
        "response.content_part.added",
    )
    sequence += 1

    final_usage = Usage()
    async for item in events:
        if item.progress:
            yield f": moa-progress {item.progress.replace(chr(10), ' ')}\n\n"
            continue
        if item.error:
            yield _sse(
                {
                    "type": "response.failed",
                    "sequence_number": sequence,
                    "response": {
                        **initial,
                        "status": "failed",
                        "error": {"code": "upstream_error", "message": item.error},
                    },
                },
                "response.failed",
            )
            return
        if item.content is not None:
            text += item.content
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": item.content,
                    "logprobs": [],
                },
                "response.output_text.delta",
            )
            sequence += 1
        if item.done and item.usage:
            final_usage = item.usage

    final_part = {**part, "text": text}
    yield _sse(
        {
            "type": "response.output_text.done",
            "sequence_number": sequence,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        },
        "response.output_text.done",
    )
    sequence += 1
    yield _sse(
        {
            "type": "response.content_part.done",
            "sequence_number": sequence,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": final_part,
        },
        "response.content_part.done",
    )
    sequence += 1
    final_item = {**output_item, "status": "completed", "content": [final_part]}
    yield _sse(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": 0,
            "item": final_item,
        },
        "response.output_item.done",
    )
    sequence += 1
    completion = Completion(text, public_model, usage=final_usage)
    final_response = responses_object(
        completion,
        public_model,
        response_id=response_id,
        message_id=message_id,
    )
    yield _sse(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": final_response,
        },
        "response.completed",
    )
