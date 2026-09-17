from __future__ import annotations

from threading import Lock
from typing import Any


class InMemoryEventSink:
    """Test/demo sink. Persistent storage adapters are introduced after Phase 1."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(dict(event))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]
