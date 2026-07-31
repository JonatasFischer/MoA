from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from moa_gateway.domain import Completion, StreamEvent
from moa_gateway.protocols import chat_completion, chat_stream, parse_chat_request


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
