from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from moa_gateway.domain import Completion, StreamEvent
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


TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

TOOL_CALL = {
    "id": "call_123",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
}


def test_chat_parser_preserves_tools_calls_and_results() -> None:
    request = parse_chat_request(
        {
            "model": "moa-code",
            "messages": [
                {"role": "user", "content": "Read the README"},
                {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Project documentation",
                },
            ],
            "tools": [TOOL],
            "tool_choice": "auto",
        }
    )

    assert request.tools == [TOOL]
    assert request.tool_choice == "auto"
    assert request.messages[1]["tool_calls"] == [TOOL_CALL]
    assert request.messages[2]["tool_call_id"] == "call_123"


def test_chat_completion_emits_tool_calls() -> None:
    payload = chat_completion(
        Completion(
            content="",
            model="local",
            finish_reason="tool_calls",
            tool_calls=[TOOL_CALL],
        ),
        "moa-code",
    )

    choice = payload["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [TOOL_CALL]
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_chat_stream_preserves_tool_argument_fragments() -> None:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":"},
                }
            ]
        )
        yield StreamEvent(
            tool_calls=[
                {"index": 0, "function": {"arguments": "\"README.md\"}"}}
            ]
        )
        yield StreamEvent(finish_reason="tool_calls", done=True)

    chunks = "".join([chunk async for chunk in chat_stream(events(), "moa-code")])
    payloads = [
        json.loads(line[6:])
        for line in chunks.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [
        payload["choices"][0]["delta"]["tool_calls"]
        for payload in payloads
        if payload.get("choices")
        and payload["choices"][0]["delta"].get("tool_calls")
    ]

    assert deltas[0][0]["function"]["arguments"] == '{"path":'
    assert deltas[1][0]["function"]["arguments"] == '"README.md"}'


def test_anthropic_tools_calls_and_results_round_trip() -> None:
    request = parse_anthropic_request(
        {
            "model": "claude-moa-code",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Read the README"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "read_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Project documentation",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": TOOL["function"]["parameters"],
                }
            ],
            "tool_choice": {"type": "tool", "name": "read_file"},
        }
    )

    assert request.tools == [TOOL]
    assert request.tool_choice == {
        "type": "function",
        "function": {"name": "read_file"},
    }
    assert request.messages[1]["tool_calls"] == [TOOL_CALL]
    assert request.messages[2] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Project documentation",
    }

    payload = anthropic_message(
        Completion("", "local", finish_reason="tool_calls", tool_calls=[TOOL_CALL]),
        "claude-moa-code",
    )
    assert payload["content"] == [
        {
            "type": "tool_use",
            "id": "call_123",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    ]
    assert payload["stop_reason"] == "tool_use"


def test_responses_tools_calls_and_results_round_trip() -> None:
    request = parse_responses_request(
        {
            "model": "moa-code",
            "input": [
                {"type": "message", "role": "user", "content": "Read it"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Project documentation",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": TOOL["function"]["parameters"],
                }
            ],
            "tool_choice": {"type": "function", "name": "read_file"},
        }
    )

    assert request.tools == [TOOL]
    assert request.messages[1]["tool_calls"] == [TOOL_CALL]
    assert request.messages[2]["tool_call_id"] == "call_123"

    payload = responses_object(
        Completion("", "local", finish_reason="tool_calls", tool_calls=[TOOL_CALL]),
        "moa-code",
    )
    assert payload["output"] == [
        {
            "id": payload["output"][0]["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": "call_123",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streamer", "expected"),
    [
        (anthropic_stream, "input_json_delta"),
        (responses_stream, "response.function_call_arguments.delta"),
    ],
)
async def test_native_streams_emit_tool_argument_deltas(streamer, expected) -> None:
    async def events() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":'},
                }
            ]
        )
        yield StreamEvent(
            tool_calls=[
                {"index": 0, "function": {"arguments": '"README.md"}'}}
            ]
        )
        yield StreamEvent(finish_reason="tool_calls", done=True)

    chunks = "".join([chunk async for chunk in streamer(events(), "moa-code")])

    assert expected in chunks
    assert "call_123" in chunks
    assert "README.md" in chunks
