#!/usr/bin/env python3
"""Executable Phase 9 release-integration gate. Hard cutoff 9.9, critical defects fail."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = 9.9


def run(name: str, args: list[str], timeout: int = 240) -> dict:
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {"name": name, "passed": p.returncode == 0, "returncode": p.returncode, "tail": (p.stdout + p.stderr)[-6000:]}


def static_gate() -> dict:
    backend = (ROOT / "backend/server.py").read_text(encoding="utf-8")
    api = (ROOT / "waf/api/production_api.py").read_text(encoding="utf-8")
    store = (ROOT / "waf/storage/production.py").read_text(encoding="utf-8")
    migration = (ROOT / "supabase/migrations/20260917150000_phase9_runtime_security.sql").read_text(encoding="utf-8")
    checks = {
        "secure_fastapi_entrypoint": "create_app()" in backend and "reload=False" in backend,
        "bearer_auth_dependency": "parse_bearer_token" in api and 'Depends(principal)' in api,
        "explicit_permissions": "approve:rules" in api and "manage:deployments" in api,
        "no_wildcard_cors": 'allow_origins=["*"]' not in api and 'allow_headers=["*"]' not in api,
        "server_only_supabase_role": "SUPABASE_SERVICE_ROLE_KEY" not in backend and "service-role" in store,
        "raw_payload_excluded": '"payload": None' in store and '"body": None' in store and '"headers": {}' in store,
        "source_identity_hashed": "sha256" in store and "hash_identifier" in store,
        "legacy_anon_revoked": "REVOKE ALL ON public.threats FROM anon" in migration and "REVOKE ALL ON public.request_logs FROM anon" in migration,
        "legacy_raw_cleanup": "UPDATE public.threats SET payload = NULL" in migration and "UPDATE public.request_logs" in migration,
        "future_privacy_checks": "phase9_request_body_null" in migration and "phase9_threat_source_ip_hash" in migration,
    }
    return {"name": "phase9-static-contract", "passed": all(checks.values()), "checks": checks}


def scenario_gate() -> dict:
    result = run("challenge-scenario-benchmark", [sys.executable, "scripts/phase9_scenario_benchmark.py"], timeout=240)
    if not result["passed"]:
        return result
    data = json.loads((ROOT / "phase9_scenario_evidence.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in data["scenarios"]}
    checks = {
        "baseline_allow": by_name["baseline"]["decision"] == "allow",
        "https_traffic_inspected": by_name["https_termination"]["evidence_schema"] == "evidence-v1",
        "known_sql_blocked": by_name["known_sql"]["decision"] == "block",
        "known_xss_blocked": by_name["known_xss"]["decision"] == "block",
        "zero_day_variant_blocked": by_name["zero_day_variant"]["decision"] == "block",
        "api_abuse_behaviour_signal": data["api_abuse"]["behavioural_escalation_observed"] is True,
        "privacy_safe": all(v is False for v in data["privacy_contract"].values()),
        "performance_measured": data["performance"]["requests"] == 500 and data["performance"]["throughput_requests_per_second"] > 0,
    }
    return {"name": "challenge-scenario-gate", "passed": all(checks.values()), "checks": checks, "performance": data["performance"]}


def main() -> int:
    checks = [
        run("full-regression", [sys.executable, "-m", "pytest", "-q"], timeout=300),
        run("compileall", [sys.executable, "-m", "compileall", "-q", "waf", "tests", "scripts"], timeout=120),
        run("phase8-security-continuity", [sys.executable, "-m", "pytest", "-q", "tests/test_phase8_security.py"], timeout=120),
        run("phase9-api-tests", [sys.executable, "-m", "pytest", "-q", "tests/test_phase9_production_api.py"], timeout=180),
        scenario_gate(),
        run("tls-nginx-smoke", [sys.executable, "scripts/phase9_tls_smoke.py"], timeout=180),
        static_gate(),
    ]
    critical = [c["name"] for c in checks if not c.get("passed")]
    score = 10.0 if not critical else max(0.0, 10.0 - 1.0 * len(critical))
    result = {
        "phase": 9,
        "status": "PASS" if score >= CUTOFF and not critical else "FAIL",
        "score": score,
        "cutoff": CUTOFF,
        "critical_defects": critical,
        "checks": checks,
        "completed": [
            "production FastAPI authentication/RBAC wiring",
            "server-only Supabase REST adapter",
            "privacy-safe runtime telemetry with hashed source identity",
            "legacy Supabase anonymous-access lockdown and raw-material cleanup migration",
            "local HTTPS/TLS termination integration through nginx",
            "Challenge 3 deterministic scenarios and load/latency evidence",
        ],
        "open": [
            "live Supabase migration/application against a real project",
            "external certificate issuance/rotation and public HTTPS verification",
            "ModSecurity/Coraza runtime module integration (module absent in verification environment)",
            "Internet-scale distributed load/failure testing",
            "authenticated dashboard UX and five-minute submission recording",
            "final technical report and presentation",
        ],
        "honesty_boundary": "Phase 9 proves repository-local production integration and controlled local HTTPS/scenario behaviour. It does not claim live cloud infrastructure, external certificate management, ModSecurity/Coraza installation, or Internet-scale performance.",
    }
    (ROOT / "phase9_master_exam_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
