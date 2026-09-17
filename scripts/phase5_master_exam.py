from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    commands = [
        ("full-regression", [sys.executable, "-m", "pytest", "-q"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q", "waf", "tests"]),
        ("phase5-evidence", [sys.executable, "-m", "pytest", "-q", "tests/test_phase5_explainability.py"]),
        ("explanation-overhead", [sys.executable, "phase5_explainability_benchmark.py"]),
    ]
    for name, command in commands:
        ok, output = run(command)
        checks.append((name, ok, output[-1800:]))

    # Critical privacy scan: no Phase 5 evidence implementation may set raw-retention flags to true.
    explainability = (ROOT / "waf" / "explainability.py").read_text(encoding="utf-8")
    migration = (ROOT / "supabase" / "migrations" / "20260917101500_phase5_decision_evidence.sql").read_text(encoding="utf-8")
    privacy_ok = (
        '"raw_payload_retained": False' in explainability
        and '"raw_query_retained": False' in explainability
        and '"raw_headers_retained": False' in explainability
        and "payload text" not in migration.lower()
        and "query_params" not in migration.lower()
        and "headers jsonb" not in migration.lower()
        and "source_ip" not in migration.lower()
    )
    checks.append(("privacy-static-gate", privacy_ok, "Phase 5 evidence + decision_evidence migration checked for raw payload/query/header/source-IP storage"))

    passed = sum(1 for _, ok, _ in checks if ok)
    critical = [name for name, ok, _ in checks if not ok]
    score = round((passed / len(checks)) * 10.0, 2)
    report = {
        "phase": 5,
        "score": score,
        "cutoff": 9.9,
        "critical_defects": critical,
        "checks": [{"name": n, "passed": ok, "tail": text} for n, ok, text in checks],
        "evaluation_scope": "repository regression, deterministic evidence tests, privacy-static gate and explanation-overhead benchmark",
    }
    print(json.dumps(report, indent=2))
    return 0 if score >= 9.9 and not critical else 1


if __name__ == "__main__":
    raise SystemExit(main())
