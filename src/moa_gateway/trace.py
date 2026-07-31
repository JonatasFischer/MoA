from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceRecorder:
    def __init__(
        self,
        path: str | None,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        self.path = Path(path) if path else None
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        self._parents: dict[str, str] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, request_id: str, **fields: Any) -> None:
        if not self.path:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "event": event,
            **fields,
        }
        if parent_request_id := self._parents.get(request_id):
            payload["parent_request_id"] = parent_request_id
        line = self._bounded_line(payload)
        line_bytes = len((line + "\n").encode("utf-8"))
        with self._lock:
            if (
                self.path.exists()
                and self.path.stat().st_size + line_bytes > self.max_bytes
            ):
                self._rotate()
            trace = self.path.open("a", encoding="utf-8")
            try:
                trace.write(line + "\n")
            finally:
                trace.close()

    def bind_parent(self, request_id: str, parent_request_id: str | None) -> None:
        if parent_request_id:
            self._parents[request_id] = parent_request_id

    def clear_parent(self, request_id: str) -> None:
        self._parents.pop(request_id, None)

    def _bounded_line(self, payload: dict[str, Any]) -> str:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        size = len((line + "\n").encode("utf-8"))
        if size <= self.max_bytes:
            return line

        bounded = dict(payload)
        bounded["truncated"] = True
        bounded["original_bytes"] = size
        for field in (
            "messages",
            "content",
            "tool_calls",
            "response_format",
            "error",
        ):
            if field in bounded:
                bounded[field] = "<truncated>"
        line = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
        if len((line + "\n").encode("utf-8")) <= self.max_bytes:
            return line
        minimal = {
            "timestamp": payload["timestamp"],
            "request_id": payload["request_id"],
            "event": payload["event"],
            "truncated": True,
            "original_bytes": size,
        }
        return json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))

    def _rotate(self) -> None:
        if not self.path or not self.path.exists():
            return
        if self.backup_count == 0:
            self.path.unlink()
            return
        oldest = Path(f"{self.path}.{self.backup_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = Path(f"{self.path}.{index}")
            if source.exists():
                source.replace(Path(f"{self.path}.{index + 1}"))
        self.path.replace(Path(f"{self.path}.1"))
