#!/usr/bin/env python3
"""Deterministic Challenge-3 scenario and performance evidence for Phase 9."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waf.core.config import WAFConfig
from waf.core.models import Decision, RequestEnvelope
from waf.edge.pipeline import EdgeWAF


def analyze(waf: EdgeWAF, name: str, method: str, scheme: str, path: str, query: str = "", body: bytes = b"", ip: str = "198.51.100.10", timestamp: float = 1_700_000_000.0) -> dict:
    request = RequestEnvelope(
        request_id=f"p9-{name}", method=method, scheme=scheme, host="example.test",
        path=path, query=query, headers={"User-Agent": "phase9-scenario"}, body=body,
        source_ip=ip, timestamp=timestamp,
    )
    result = waf.analyze(request)
    return {
        "name": name,
        "scheme": scheme,
        "decision": result.decision.value,
        "risk_score": result.risk_score,
        "reasons": list(result.reasons),
        "rule_ids": list(result.rule_ids),
        "evidence_schema": result.evidence.schema_version if result.evidence else None,
        "privacy": dict(result.evidence.privacy) if result.evidence else {},
    }


def main() -> int:
    waf = EdgeWAF(WAFConfig())
    scenarios = [
        analyze(waf, "baseline", "GET", "http", "/api/products", "page=1"),
        analyze(waf, "https_termination", "GET", "https", "/api/profile", "id=42"),
        analyze(waf, "known_sql", "GET", "https", "/search", "q=UNION SELECT password FROM users", ip="203.0.113.10"),
        analyze(waf, "known_xss", "POST", "https", "/comment", body=b"<script>alert(1)</script>", ip="203.0.113.11"),
        analyze(waf, "zero_day_variant", "POST", "https", "/api/export", body=b"$(curl http://127.0.0.1/x)", ip="203.0.113.12"),
    ]

    burst = []
    for i in range(120):
        row = analyze(waf, f"bot-{i}", "GET", "https", f"/api/item/{i % 20}", f"page={i}", ip="192.0.2.40", timestamp=1_700_001_000.0 + i * 0.05)
        burst.append(row)
    burst_block_or_alert = sum(1 for row in burst if row["decision"] != "allow")
    burst_behaviour_reason = any(any("behavioural anomaly" in reason.lower() for reason in row["reasons"]) for row in burst)

    latencies_ms = []
    started = time.perf_counter()
    for i in range(500):
        req = RequestEnvelope(
            request_id=f"perf-{i}", method="GET", scheme="https", host="example.test",
            path=f"/api/item/{i % 50}", query=f"page={i}",
            headers={"User-Agent": "phase9-perf"}, source_ip="198.51.100.55", timestamp=1_700_100_000 + i,
        )
        t0 = time.perf_counter()
        waf.analyze(req)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - started
    throughput = 500 / max(elapsed, 1e-9)
    ordered = sorted(latencies_ms)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    p99 = ordered[int(len(ordered) * 0.99) - 1]

    output = {
        "phase": 9,
        "scope": "deterministic local Challenge 3 scenarios",
        "scenarios": scenarios,
        "api_abuse": {
            "requests": len(burst),
            "non_allow_decisions": burst_block_or_alert,
            "behavioural_escalation_observed": burst_behaviour_reason,
            "policy_outcome_note": "behavioural signal is observable while combined policy may remain allow at the configured threshold",
        },
        "performance": {
            "requests": 500,
            "throughput_requests_per_second": round(throughput, 3),
            "mean_latency_ms": round(statistics.mean(latencies_ms), 4),
            "p95_latency_ms": round(p95, 4),
            "p99_latency_ms": round(p99, 4),
            "min_latency_ms": round(min(latencies_ms), 4),
            "max_latency_ms": round(max(latencies_ms), 4),
            "measurement": "in-process EdgeWAF.analyze; excludes network/TLS/upstream time",
        },
        "privacy_contract": {
            "raw_payload_retained": False,
            "raw_query_retained": False,
            "raw_headers_retained": False,
        },
        "limitations": [
            "local deterministic workload only",
            "does not claim Internet-scale throughput",
            "TLS evidence covers post-termination HTTPS input, not an externally deployed certificate",
            "ModSecurity/Coraza binary module not present in this environment",
        ],
    }
    path = ROOT / "phase9_scenario_evidence.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
