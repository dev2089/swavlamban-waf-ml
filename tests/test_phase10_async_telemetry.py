import time

from waf.storage.async_telemetry import AsyncTelemetryDispatcher


class Store:
    def __init__(self):
        self.decisions = []
        self.audits = []

    def record_decision_view(self, **payload):
        self.decisions.append(("view", payload["request_id"]))

    def record_decision(self, **payload):
        self.decisions.append(("persist", payload["request_id"]))

    def record_audit(self, **payload):
        self.audits.append(payload["request_id"])


def test_async_dispatcher_is_non_blocking_and_flushes():
    store = Store()
    dispatcher = AsyncTelemetryDispatcher(store, max_queue=4)
    assert dispatcher.enqueue_decision(request_id="req-000001", source_ip="127.0.0.1", method="GET", uri="/", result={"threat_detected": False, "blocked": False, "risk_score": 0.0})
    assert dispatcher.enqueue_audit(actor="op", action="test", target="req-000001", outcome="allow", request_id="audit-000001")
    assert dispatcher.flush(2.0)
    assert ("view", "req-000001") in store.decisions
    assert ("persist", "req-000001") in store.decisions
    assert store.audits == ["audit-000001"]
    dispatcher.close()


def test_async_dispatcher_records_backpressure():
    class SlowStore(Store):
        def record_decision(self, **payload):
            time.sleep(0.02)
            super().record_decision(**payload)

    store = SlowStore()
    dispatcher = AsyncTelemetryDispatcher(store, max_queue=1)
    accepted = 0
    for idx in range(50):
        accepted += dispatcher.enqueue_decision(request_id=f"req-{idx:06d}", source_ip=None, method="GET", uri="/", result={"threat_detected": False, "blocked": False, "risk_score": 0.0})
    dispatcher.flush(3.0)
    stats = dispatcher.stats()
    dispatcher.close()
    assert accepted > 0
    assert stats["dropped"] > 0
