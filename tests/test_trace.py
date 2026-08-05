from __future__ import annotations

import json

from moa_gateway.trace import TraceRecorder


def test_trace_rotates_and_bounds_backups(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(str(path), max_bytes=1024, backup_count=2)

    for index in range(20):
        recorder.record("model_completed", str(index), content="x" * 200)

    assert path.exists()
    assert (tmp_path / "trace.jsonl.1").exists()
    assert (tmp_path / "trace.jsonl.2").exists()
    assert not (tmp_path / "trace.jsonl.3").exists()
    for trace_path in tmp_path.glob("trace.jsonl*"):
        assert trace_path.stat().st_size <= 1024
        for line in trace_path.read_text().splitlines():
            json.loads(line)


def test_trace_truncates_single_oversized_record(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(str(path), max_bytes=1024, backup_count=1)

    recorder.record("request_started", "request", messages=[{"content": "x" * 5000}])

    record = json.loads(path.read_text())
    assert record["truncated"] is True
    assert record["messages"] == "<truncated>"
    assert path.stat().st_size <= 1024


def test_trace_subscribers_receive_events_without_a_log_path() -> None:
    recorder = TraceRecorder(None)
    events: list[dict] = []
    subscriber = events.append
    recorder.subscribe("request", subscriber)

    recorder.record("stage_started", "request", stage="filter")
    recorder.unsubscribe("request", subscriber)
    recorder.record("stage_completed", "request", stage="filter")

    assert [event["event"] for event in events] == ["stage_started"]
    assert events[0]["stage"] == "filter"
