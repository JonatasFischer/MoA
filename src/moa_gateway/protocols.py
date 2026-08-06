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


def _collapse_blocks(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(block.get("type") == "text" for block in blocks):
        return "".join(str(block.get("text") or "") for block in blocks)
    return blocks


def _canonical_content(content: Any, protocol: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail="message content must be a list or text")
    blocks: list[dict[str, Any]] = []
    text_types = {
        "chat": {"text"},
        "anthropic": {"text"},
        "responses": {"input_text", "output_text", "text"},
    }[protocol]
    for block in content:
        if not isinstance(block, dict):
            raise HTTPException(status_code=400, detail="content blocks must be objects")
        block_type = block.get("type")
        if block_type in text_types:
            text = block.get("text")
            if not isinstance(text, str):
                raise HTTPException(status_code=400, detail="text block is missing text")
            blocks.append({"type": "text", "text": text})
            continue
        if protocol == "chat" and block_type == "image_url":
            image = block.get("image_url")
            if not isinstance(image, (str, dict)):
                raise HTTPException(status_code=400, detail="image_url block is invalid")
            blocks.append({"type": "image_url", "image_url": image})
            continue
        if protocol == "anthropic" and block_type == "image":
            source = block.get("source")
            if not isinstance(source, dict):
                raise HTTPException(status_code=400, detail="image source is invalid")
            if source.get("type") == "base64":
                media_type = source.get("media_type")
                data = source.get("data")
                if not isinstance(media_type, str) or not isinstance(data, str):
                    raise HTTPException(status_code=400, detail="base64 image is invalid")
                url = f"data:{media_type};base64,{data}"
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                url = source["url"]
            else:
                raise HTTPException(status_code=400, detail="unsupported image source")
            blocks.append({"type": "image_url", "image_url": {"url": url}})
            continue
        if protocol == "responses" and block_type == "input_image":
            url = block.get("image_url")
            if not isinstance(url, str) or not url:
                raise HTTPException(status_code=400, detail="input_image needs image_url")
            image: dict[str, Any] = {"url": url}
            if block.get("detail") is not None:
                image["detail"] = block["detail"]
            blocks.append({"type": "image_url", "image_url": image})
            continue
        if protocol in {"chat", "responses"} and block_type in {"file", "input_file"}:
            file_value = block.get("file") if protocol == "chat" else {
                key: block[key]
                for key in ("file_data", "file_id", "file_url", "filename")
                if block.get(key) is not None
            }
            if not isinstance(file_value, dict) or not file_value:
                raise HTTPException(status_code=400, detail="file block is invalid")
            blocks.append({"type": "file", "file": file_value})
            continue
        if protocol == "anthropic" and block_type == "document":
            source = block.get("source")
            if not isinstance(source, dict):
                raise HTTPException(status_code=400, detail="document source is invalid")
            file_value: dict[str, Any] = {}
            if source.get("type") == "base64":
                media_type = source.get("media_type")
                data = source.get("data")
                if not isinstance(media_type, str) or not isinstance(data, str):
                    raise HTTPException(status_code=400, detail="base64 document is invalid")
                file_value["file_data"] = f"data:{media_type};base64,{data}"
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                file_value["file_url"] = source["url"]
            else:
                raise HTTPException(status_code=400, detail="unsupported document source")
            if isinstance(block.get("title"), str):
                file_value["filename"] = block["title"]
            blocks.append({"type": "file", "file": file_value})
            continue
        raise HTTPException(status_code=501, detail=f"unsupported {protocol} content block: {block_type}")
    return _collapse_blocks(blocks)


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
            "content": _canonical_content(content, "chat"),
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
    messages: list[dict[str, Any]] = []
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
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="message content must be a list or text")
        if role == "assistant":
            if not all(isinstance(block, dict) for block in content):
                raise HTTPException(status_code=400, detail="content blocks must be objects")
            ordinary = [block for block in content if block.get("type") != "tool_use"]
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    raise HTTPException(status_code=400, detail="content blocks must be objects")
                if block.get("type") != "tool_use":
                    continue
                if not all(isinstance(block.get(key), str) and block.get(key) for key in ("id", "name")):
                    raise HTTPException(status_code=400, detail="tool_use block is invalid")
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input") or {}, separators=(",", ":")
                            ),
                        },
                    }
                )
            normalized = {
                "role": "assistant",
                "content": _canonical_content(ordinary, "anthropic"),
            }
            if tool_calls:
                normalized["tool_calls"] = tool_calls
            messages.append(normalized)
            continue

        ordinary: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise HTTPException(status_code=400, detail="content blocks must be objects")
            if block.get("type") != "tool_result":
                ordinary.append(block)
                continue
            if ordinary:
                messages.append(
                    {"role": "user", "content": _canonical_content(ordinary, "anthropic")}
                )
                ordinary = []
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id:
                raise HTTPException(status_code=400, detail="tool_result needs tool_use_id")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _canonical_content(block.get("content", ""), "anthropic"),
                }
            )
        if ordinary:
            messages.append(
                {"role": "user", "content": _canonical_content(ordinary, "anthropic")}
            )

    tools: list[dict[str, Any]] = []
    for tool in data.get("tools") or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise HTTPException(status_code=400, detail="invalid Anthropic tool")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    choice = data.get("tool_choice")
    if isinstance(choice, dict):
        choice_type = choice.get("type")
        if choice_type == "tool":
            choice = {"type": "function", "function": {"name": choice.get("name")}}
        else:
            choice = {"auto": "auto", "any": "required", "none": "none"}.get(
                choice_type, choice
            )

    return CanonicalRequest(
        requested_model=data.get("model"),
        messages=messages,
        max_tokens=data.get("max_tokens"),
        temperature=data.get("temperature"),
        stop=data.get("stop_sequences"),
        tools=tools,
        tool_choice=choice,
    )


def parse_responses_request(body: Any) -> CanonicalRequest:
    data = _require_object(body)
    if data.get("previous_response_id") or data.get("conversation"):
        raise HTTPException(
            status_code=501,
            detail="server-managed Responses conversations are not implemented yet",
        )

    messages: list[dict[str, Any]] = []
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
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="Responses items must be objects")
            item_type = item.get("type", "message")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not isinstance(name, str):
                    raise HTTPException(status_code=400, detail="invalid function_call item")
                arguments = item.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                )
                continue
            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise HTTPException(status_code=400, detail="function_call_output needs call_id")
                output = item.get("output", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output if isinstance(output, str) else json.dumps(output),
                    }
                )
                continue
            if item_type != "message":
                raise HTTPException(status_code=501, detail=f"unsupported Responses item: {item_type}")
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise HTTPException(
                    status_code=400, detail=f"unsupported input role: {role}"
                )
            messages.append(
                {
                    "role": "system" if role == "developer" else role,
                    "content": _canonical_content(item.get("content", ""), "responses"),
                }
            )
    else:
        raise HTTPException(status_code=400, detail="input must be text or message items")

    tools: list[dict[str, Any]] = []
    for tool in data.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("name"), str):
            raise HTTPException(status_code=400, detail="invalid Responses function tool")
        function = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object"},
        }
        if tool.get("strict") is not None:
            function["strict"] = tool["strict"]
        tools.append({"type": "function", "function": function})
    choice = data.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        choice = {"type": "function", "function": {"name": choice.get("name")}}
    return CanonicalRequest(
        requested_model=data.get("model"),
        messages=messages,
        max_tokens=data.get("max_output_tokens"),
        temperature=data.get("temperature"),
        tools=tools,
        tool_choice=choice,
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
    content: list[dict[str, Any]] = []
    if completion.content:
        content.append({"type": "text", "text": completion.content})
    for call in completion.tool_calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="model emitted invalid tool arguments") from exc
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": function.get("name"),
                "input": arguments,
            }
        )
    return {
        "id": _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": public_model,
        "content": content,
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
        "input_tokens_details": {"cached_tokens": usage.cached_input_tokens},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_output_tokens},
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
    output: list[dict[str, Any]] = []
    if completion.content or not completion.tool_calls:
        output.append(
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
        )
    for index, call in enumerate(completion.tool_calls):
        function = call.get("function") or {}
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("id") or f"call_{index}",
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": public_model,
        "output": output,
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
    text_index: int | None = None
    tool_indexes: dict[int, int] = {}
    open_indexes: list[int] = []
    next_index = 0
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
            if text_index is None:
                text_index = next_index
                next_index += 1
                open_indexes.append(text_index)
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": text_index,
                    "delta": {"type": "text_delta", "text": item.content},
                },
                "content_block_delta",
            )
        for position, call in enumerate(item.tool_calls or []):
            source_index = call.get("index")
            if not isinstance(source_index, int):
                source_index = position
            function = call.get("function") or {}
            if source_index not in tool_indexes:
                block_index = next_index
                next_index += 1
                tool_indexes[source_index] = block_index
                open_indexes.append(block_index)
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                            "name": function.get("name") or "",
                            "input": {},
                        },
                    },
                    "content_block_start",
                )
            arguments = function.get("arguments")
            if arguments is not None:
                yield _sse(
                    {
                        "type": "content_block_delta",
                        "index": tool_indexes[source_index],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": str(arguments),
                        },
                    },
                    "content_block_delta",
                )
        if item.done:
            usage = item.usage or Usage()
            if not open_indexes:
                open_indexes.append(0)
                yield _sse(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "content_block_start",
                )
            for block_index in open_indexes:
                yield _sse(
                    {"type": "content_block_stop", "index": block_index},
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
            return


async def responses_stream(
    events: AsyncIterator[StreamEvent], public_model: str
) -> AsyncIterator[str]:
    response_id = _id("resp")
    message_id = _id("msg")
    sequence = 0

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
    final_usage = Usage()
    text = ""
    text_index: int | None = None
    text_item: dict[str, Any] | None = None
    text_part = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
    tools: dict[int, dict[str, Any]] = {}
    next_output_index = 0
    saw_done = False
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
            if text_item is None:
                text_index = next_output_index
                next_output_index += 1
                text_item = {
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
                        "output_index": text_index,
                        "item": text_item,
                    },
                    "response.output_item.added",
                )
                sequence += 1
                yield _sse(
                    {
                        "type": "response.content_part.added",
                        "sequence_number": sequence,
                        "item_id": message_id,
                        "output_index": text_index,
                        "content_index": 0,
                        "part": text_part,
                    },
                    "response.content_part.added",
                )
                sequence += 1
            text += item.content
            yield _sse(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": message_id,
                    "output_index": text_index,
                    "content_index": 0,
                    "delta": item.content,
                    "logprobs": [],
                },
                "response.output_text.delta",
            )
            sequence += 1
        for position, call in enumerate(item.tool_calls or []):
            source_index = call.get("index")
            if not isinstance(source_index, int):
                source_index = position
            function = call.get("function") or {}
            state = tools.get(source_index)
            if state is None:
                output_index = next_output_index
                next_output_index += 1
                state = {
                    "output_index": output_index,
                    "id": f"fc_{uuid.uuid4().hex}",
                    "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                    "name": function.get("name") or "",
                    "arguments": "",
                }
                tools[source_index] = state
                function_item = {
                    "id": state["id"],
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": state["call_id"],
                    "name": state["name"],
                    "arguments": "",
                }
                yield _sse(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": sequence,
                        "output_index": output_index,
                        "item": function_item,
                    },
                    "response.output_item.added",
                )
                sequence += 1
            if call.get("id"):
                state["call_id"] = call["id"]
            if function.get("name"):
                state["name"] = function["name"]
            if function.get("arguments") is not None:
                delta = str(function["arguments"])
                state["arguments"] += delta
                yield _sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence,
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": delta,
                    },
                    "response.function_call_arguments.delta",
                )
                sequence += 1
        if item.done:
            saw_done = True
            if item.usage:
                final_usage = item.usage
            break

    if not saw_done:
        yield _sse(
            {
                "type": "response.failed",
                "sequence_number": sequence,
                "response": {
                    **initial,
                    "status": "failed",
                    "error": {
                        "code": "upstream_error",
                        "message": "upstream stream ended without a terminal event",
                    },
                },
            },
            "response.failed",
        )
        return

    final_output: list[tuple[int, dict[str, Any]]] = []
    if text_item is not None and text_index is not None:
        final_part = {**text_part, "text": text}
        yield _sse(
            {
                "type": "response.output_text.done",
                "sequence_number": sequence,
                "item_id": message_id,
                "output_index": text_index,
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
                "output_index": text_index,
                "content_index": 0,
                "part": final_part,
            },
            "response.content_part.done",
        )
        sequence += 1
        final_item = {**text_item, "status": "completed", "content": [final_part]}
        yield _sse(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": text_index,
                "item": final_item,
            },
            "response.output_item.done",
        )
        sequence += 1
        final_output.append((text_index, final_item))
    for source_index in sorted(tools):
        state = tools[source_index]
        final_item = {
            "id": state["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": state["call_id"],
            "name": state["name"],
            "arguments": state["arguments"],
        }
        yield _sse(
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": sequence,
                "item_id": state["id"],
                "output_index": state["output_index"],
                "arguments": state["arguments"],
            },
            "response.function_call_arguments.done",
        )
        sequence += 1
        yield _sse(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": state["output_index"],
                "item": final_item,
            },
            "response.output_item.done",
        )
        sequence += 1
        final_output.append((state["output_index"], final_item))
    if not final_output:
        final_output.append(
            (
                0,
                {
                    "id": message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{**text_part, "text": ""}],
                },
            )
        )
    completion = Completion(text, public_model, usage=final_usage)
    final_response = responses_object(
        completion,
        public_model,
        response_id=response_id,
        message_id=message_id,
    )
    final_response["output"] = [item for _, item in sorted(final_output)]
    yield _sse(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": final_response,
        },
        "response.completed",
    )
