"""Bounded asynchronous telemetry dispatcher for non-blocking request paths."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class _Task:
    kind: str
    payload: Mapping[str, Any]


class AsyncTelemetryDispatcher:
    """Move persistence work off the request path with explicit backpressure metrics."""

    def __init__(self, store: Any, max_queue: int = 256) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        self.store = store
        self.queue: Queue[_Task | None] = Queue(maxsize=max_queue)
        self.stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._dropped = 0
        self._failed = 0
        self._last_error = ""

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.stop.clear()
        self._thread = Thread(target=self._worker, name="waf-telemetry", daemon=True)
        self._thread.start()

    def enqueue_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> bool:
        return self._enqueue(_Task("decision", {
            "request_id": request_id,
            "source_ip": source_ip,
            "method": method,
            "uri": uri,
            "result": dict(result),
        }))

    def enqueue_audit(self, *, actor: str, action: str, target: str, outcome: str, request_id: str) -> bool:
        return self._enqueue(_Task("audit", {
            "actor": actor,
            "action": action,
            "target": target,
            "outcome": outcome,
            "request_id": request_id,
        }))

    def _enqueue(self, task: _Task) -> bool:
        self._ensure_started()
        try:
            self.queue.put_nowait(task)
            return True
        except Full:
            with self._lock:
                self._dropped += 1
            return False

    def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                task = self.queue.get(timeout=0.25)
            except Exception:
                continue
            if task is None:
                self.queue.task_done()
                break
            try:
                if task.kind == "decision":
                    self.store.record_decision(**task.payload)
                elif task.kind == "audit":
                    self.store.record_audit(**task.payload)
                else:  # pragma: no cover - internal invariant guard
                    raise RuntimeError("unknown telemetry task")
            except Exception as exc:  # store outage must not break WAF enforcement
                with self._lock:
                    self._failed += 1
                    self._last_error = type(exc).__name__
            finally:
                self.queue.task_done()

    def close(self, timeout: float = 2.0) -> None:
        self.stop.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
            self._thread.join(timeout=timeout)

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "queued": self.queue.qsize(),
                "dropped": self._dropped,
                "failed": self._failed,
                "last_error_type": self._last_error,
            }
