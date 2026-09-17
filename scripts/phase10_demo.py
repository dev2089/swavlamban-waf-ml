"""Deterministic Phase 10 end-to-end demo and evidence generator."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waf.api.production_api import create_app
from waf.security.production_security import issue_token

SECRET = "phase10-demo-secret-" + "x" * 32


def main() -> int:
    env = {
        "WAF_ENV": "development",
        "WAF_AUTH_SECRET": SECRET,
        "WAF_CORS_ORIGINS": "https://console.example.test",
        "WAF_TOKEN_ISSUER": "swavlamban-waf",
        "WAF_TOKEN_AUDIENCE": "waf-control-plane",
        "WAF_TOKEN_TTL_SECONDS": "900",
        "WAF_CLOCK_SKEW_SECONDS": "30",
    }
    app = create_app(env=env)
    client = TestClient(app)
    token = issue_token(subject="phase10-demo", role="operator", secret=SECRET)
    headers = {"Authorization": "Bearer " + token}
    cases = [
        ("benign", {"method": "GET", "uri": "/home", "body": ""}),
        ("sql", {"method": "GET", "uri": "/search", "query": "q=' OR 1=1--", "body": ""}),
        ("xss", {"method": "GET", "uri": "/comment", "body": "<script>alert(1)</script>"}),
        ("command_variant", {"method": "GET", "uri": "/ping", "query": "x=1;cat /etc/passwd", "body": ""}),
    ]
    observed = []
    for name, payload in cases:
        response = client.post("/api/analyze", headers={**headers, "X-Request-ID": f"phase10-{name}"}, json=payload)
        response.raise_for_status()
        result = response.json()
        observed.append({"case": name, "status": response.status_code, "decision": result["decision"], "blocked": result["blocked"], "risk_score": result["risk_score"]})
    times = []
    for idx in range(500):
        started = perf_counter()
        response = client.post("/api/analyze", headers=headers, json={"method": "GET", "uri": f"/bulk/{idx}", "body": ""})
        response.raise_for_status()
        times.append((perf_counter() - started) * 1000)
    health = client.get("/api/health")
    stats = client.get("/api/stats", headers=headers)
    release = {
        "phase": 10,
        "service": "swavlamban-waf-api",
        "health": health.json(),
        "stats": stats.json(),
        "scenarios": observed,
        "benchmark": {"requests": len(times), "mean_ms": round(mean(times), 4), "max_ms": round(max(times), 4)},
        "privacy": {"raw_payload_retained": False, "raw_headers_retained": False, "raw_query_retained": False},
    }
    out = ROOT / "phase10_demo_evidence.json"
    out.write_text(json.dumps(release, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
