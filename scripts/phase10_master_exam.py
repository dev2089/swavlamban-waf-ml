"""Strict Phase 10 release gate with bounded subprocesses and explicit diagnostics."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str], env: dict[str, str] | None = None, timeout: float = 180.0) -> dict[str, object]:
    try:
        proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
        return {
            "id": label,
            "command": " ".join(command),
            "result": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "timeout_seconds": timeout,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "id": label,
            "command": " ".join(command),
            "result": "FAIL",
            "returncode": 124,
            "timeout_seconds": timeout,
            "stdout": (exc.stdout or "")[-8000:],
            "stderr": ((exc.stderr or "") if isinstance(exc.stderr, str) else str(exc.stderr or ""))[-8000:],
            "diagnostic": "subprocess timeout",
        }


def wait(url: str, timeout: float = 90.0, process: subprocess.Popen[str] | None = None, log_paths: list[Path] | None = None) -> None:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            tails = []
            for path in log_paths or []:
                if path.exists():
                    tails.append(f"--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')[-5000:]}")
            raise RuntimeError(f"service exited before readiness: {url}; returncode={process.returncode}\n" + "\n".join(tails))
        try:
            with urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last = exc
        time.sleep(0.25)
    tails = []
    for path in log_paths or []:
        if path.exists():
            tails.append(f"--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')[-5000:]}")
    raise RuntimeError(f"service not ready: {url}: {last}\n" + "\n".join(tails))


def load_json(name: str) -> tuple[dict[str, object] | None, str | None]:
    path = ROOT / name
    if not path.exists():
        return None, f"missing evidence file: {name}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"invalid JSON in {name}: {exc}"


def write_runtime_reports(checks: list[dict[str, object]]) -> None:
    handoff = ROOT / "handoff"
    demo, demo_err = load_json("phase10_demo_evidence.json")
    load, load_err = load_json("phase10_load_evidence.json")
    tls, tls_err = load_json("phase10_tls_evidence.json")
    waf, waf_err = load_json("phase10_waf_enforcement_evidence.json")
    replay, replay_err = load_json("phase10_rule_replay_evidence.json")
    errors = [x for x in (demo_err, load_err, tls_err, waf_err, replay_err) if x]

    perf_text = ["# Phase 10 Performance Benchmark", ""]
    if demo and load:
        perf_text += [
            f"Deterministic in-process demo: {demo['benchmark']['requests']} requests, mean={demo['benchmark']['mean_ms']} ms, max={demo['benchmark']['max_ms']} ms.",
            f"Bounded network harness: concurrency={load['concurrency']}, requested_rate={load['requested_rate_per_second']} req/s, attempted={load['requests_attempted']}, achieved={load['achieved_requests_per_second']} req/s, p50={load['latency_ms']['p50']} ms, p95={load['latency_ms']['p95']} ms, p99={load['latency_ms']['p99']} ms, error_rate={load['error_rate']}.",
            "These are bounded local measurements, not Internet-scale capacity claims.",
        ]
    else:
        perf_text += ["Evidence incomplete:", *[f"- {x}" for x in errors]]
    (handoff / "PHASE10_PERFORMANCE_BENCHMARK.md").write_text("\n".join(perf_text) + "\n", encoding="utf-8")

    ml_text = ["# Phase 10 ML Evaluation", "", "Runtime protection combines supervised, unsupervised and behavioural signals over versioned synthetic HTTP data. Synthetic metrics are not field-accuracy claims."]
    if replay:
        ml_text += [f"Rule replay {replay.get('rule_id')}: valid={replay.get('validation', {}).get('valid')}; positive_matched={replay.get('positive_example', {}).get('matched')}; negative_matched={replay.get('negative_example', {}).get('matched')}."]
    (handoff / "PHASE10_ML_EVALUATION.md").write_text("\n".join(ml_text) + "\n", encoding="utf-8")

    reliability = ["# Phase 10 Reliability Report", "", "Reliability coverage includes request validation, authentication/RBAC, body and response limits, rate limiting, upstream failure handling, TLS termination, ModSecurity enforcement, WebSocket authentication and the deterministic dashboard/demo path."]
    if tls and waf:
        reliability += [f"HTTPS allow={tls.get('https_allow_status')}; HTTPS SQL block={tls.get('https_sql_block_status')}; blocked reached upstream={tls.get('https_block_reached_upstream')}.", f"ModSecurity connector={waf.get('nginx_modsecurity_connector')}; SQL blocked at WAF={waf.get('sql_blocked_at_waf')}; blocked reached upstream={waf.get('blocked_request_reached_upstream')}."]
    (handoff / "PHASE10_RELIABILITY_REPORT.md").write_text("\n".join(reliability) + "\n", encoding="utf-8")


def main() -> int:
    checks: list[dict[str, object]] = []
    checks.append(run("full_regression", [sys.executable, "-m", "pytest", "-q"], timeout=300))
    checks.append(run("compile", [sys.executable, "-m", "compileall", "-q", "waf", "tests", "scripts"], timeout=60))
    checks.append(run("demo", [sys.executable, "scripts/phase10_demo.py"], timeout=180))
    checks.append(run("rule_replay", [sys.executable, "scripts/phase10_rule_replay.py"], timeout=60))
    checks.append(run("tls_e2e", [sys.executable, "scripts/phase10_tls_e2e.py"], timeout=120))

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT), "WAF_ENV": "development", "WAF_GATEWAY_HOST": "127.0.0.1",
        "WAF_GATEWAY_PORT": "18081", "WAF_UPSTREAM_URL": "http://127.0.0.1:19090", "WAF_RATE_LIMIT_PER_MINUTE": "1000",
    })
    processes: list[subprocess.Popen[str]] = []
    handles = []
    try:
        upstream_log = (ROOT / "phase10_master_upstream.log").open("w", encoding="utf-8")
        gateway_log = (ROOT / "phase10_master_gateway.log").open("w", encoding="utf-8")
        handles = [upstream_log, gateway_log]
        upstream = subprocess.Popen([sys.executable, "-m", "http.server", "19090", "--bind", "127.0.0.1", "--directory", str(ROOT)], cwd=ROOT, env=env, stdout=upstream_log, stderr=subprocess.STDOUT, text=True)
        gateway = subprocess.Popen([sys.executable, "-m", "waf.gateway.proxy"], cwd=ROOT, env=env, stdout=gateway_log, stderr=subprocess.STDOUT, text=True)
        processes.extend([upstream, gateway])
        wait("http://127.0.0.1:18081/__waf_health", process=gateway, log_paths=[ROOT / "phase10_master_upstream.log", ROOT / "phase10_master_gateway.log"])
        checks.append(run("network_load", [sys.executable, "scripts/phase10_load_harness.py", "--url", "http://127.0.0.1:18081/", "--duration", "2", "--concurrency", "20", "--rate", "100", "--payload-profile", "mixed"], timeout=120))
    except Exception as exc:
        checks.append({"id": "gateway_network", "result": "FAIL", "returncode": 1, "diagnostic": str(exc)})
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in handles:
            handle.close()

    checks.append(run("modsecurity_e2e", [sys.executable, "scripts/phase10_waf_enforcement_e2e.py"], timeout=180))
    checks.append(run("dashboard_demo", [sys.executable, "scripts/phase10_dashboard_demo.py"], timeout=480))
    checks.append(run("audit_manifest", [sys.executable, "scripts/phase10_audit.py"], timeout=120))
    write_runtime_reports(checks)

    required = [
        "phase10_demo_evidence.json", "phase10_rule_replay_evidence.json", "phase10_load_evidence.json",
        "phase10_tls_evidence.json", "phase10_waf_enforcement_evidence.json", "handoff/PHASE10_REQUIREMENT_TRACEABILITY.json",
        "handoff/PHASE10_PRODUCTION_READINESS.json", "handoff/PHASE10_SECURITY_AUDIT.md", "handoff/PHASE10_CLAIM_LEDGER.md",
        "handoff/PHASE10_NEGATIVE_EVIDENCE.md", "handoff/PHASE10_ML_EVALUATION.md", "handoff/PHASE10_PERFORMANCE_BENCHMARK.md",
        "handoff/PHASE10_RELIABILITY_REPORT.md", "artifacts/PHASE10_DEMO_VIDEO.webm",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    checks.append({"id": "required_evidence", "result": "PASS" if not missing else "FAIL", "returncode": 0 if not missing else 1, "missing": missing})

    waf, _ = load_json("phase10_waf_enforcement_evidence.json")
    tls, _ = load_json("phase10_tls_evidence.json")
    replay, _ = load_json("phase10_rule_replay_evidence.json")
    acceptance = bool(
        waf and waf.get("sql_blocked_at_waf") is True and waf.get("blocked_request_reached_upstream") is False
        and tls and tls.get("https_allow_status") == 200 and tls.get("https_sql_block_status") == 403 and tls.get("https_block_reached_upstream") is False
        and replay and replay.get("validation", {}).get("valid") is True and replay.get("positive_example", {}).get("matched") is True and replay.get("negative_example", {}).get("matched") is False
    )
    checks.append({"id": "acceptance_evidence", "result": "PASS" if acceptance else "FAIL", "returncode": 0 if acceptance else 1})

    failures = [c for c in checks if c.get("result") != "PASS"]
    critical_ids = {"full_regression", "compile", "tls_e2e", "modsecurity_e2e", "acceptance_evidence"}
    critical = [str(c.get("id")) for c in failures if c.get("id") in critical_ids]
    result = {
        "phase": 10,
        "result": "PASS" if not failures else "FAIL",
        "score": 100.0 if not failures else 0.0,
        "cutoff": 100.0,
        "critical_failures": critical,
        "failed_checks": [c.get("id") for c in failures],
        "checks": checks,
        "external_boundaries": ["public certificate issuance/rotation is not tested; local TLS only", "Internet-scale distributed capacity is not claimed", "synthetic ML evaluation is not field accuracy"],
    }
    (ROOT / "phase10_master_exam_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
