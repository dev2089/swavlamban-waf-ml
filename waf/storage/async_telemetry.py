"""Bounded asynchronous telemetry dispatcher for non-blocking request paths."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic, sleep
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
        self._closed = False
        self._thread: Thread | None = None
        self._lock = Lock()
        self._dropped = 0
        self._failed = 0
        self._last_error = ""

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("telemetry dispatcher is closed")
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._worker, name="waf-telemetry", daemon=True)
        self._thread.start()

    def enqueue_decision(self, *, request_id: str, source_ip: str | None, method: str, uri: str, result: Mapping[str, Any]) -> bool:
        task = _Task("decision", {
            "request_id": request_id,
            "source_ip": source_ip,
            "method": method,
            "uri": uri,
            "result": dict(result),
        })
        self._ensure_started()
        preview = getattr(self.store, "record_decision_view", None)
        if preview is not None:
            try:
                preview(**task.payload)
            except Exception as exc:  # local metrics must never break enforcement
                with self._lock:
                    self._failed += 1
                    self._last_error = type(exc).__name__
        return self._enqueue(task, started=True)

    def enqueue_audit(self, *, actor: str, action: str, target: str, outcome: str, request_id: str) -> bool:
        return self._enqueue(_Task("audit", {
            "actor": actor,
            "action": action,
            "target": target,
            "outcome": outcome,
            "request_id": request_id,
        }))

    def _enqueue(self, task: _Task, *, started: bool = False) -> bool:
        if not started:
            self._ensure_started()
        try:
            self.queue.put_nowait(task)
            return True
        except Full:
            with self._lock:
                self._dropped += 1
            return False

    def _worker(self) -> None:
        while True:
            try:
                task = self.queue.get(timeout=0.25)
            except Empty:
                with self._lock:
                    closed = self._closed
                if closed and self.queue.empty():
                    return
                continue
            if task is None:
                self.queue.task_done()
                return
            try:
                if task.kind == "decision":
                    self.store.record_decision(**task.payload)
                elif task.kind == "audit":
                    self.store.record_audit(**task.payload)
                else:  # pragma: no cover
                    raise RuntimeError("unknown telemetry task")
            except Exception as exc:
                with self._lock:
                    self._failed += 1
                    self._last_error = type(exc).__name__
            finally:
                self.queue.task_done()

    def flush(self, timeout: float = 2.0) -> bool:
        deadline = monotonic() + timeout
        while self.queue.unfinished_tasks and monotonic() < deadline:
            sleep(0.01)
        return self.queue.unfinished_tasks == 0

    def close(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._thread is not None and self._thread.is_alive():
            deadline = monotonic() + timeout
            remaining = max(0.0, deadline - monotonic())
            self.flush(remaining)
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
            self._thread.join(timeout=max(0.0, deadline - monotonic()))

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "queued": self.queue.qsize(),
                "dropped": self._dropped,
                "failed": self._failed,
                "last_error_type": self._last_error,
            }
