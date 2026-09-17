from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    ok, output = run([sys.executable, "-m", "pytest", "-q"])
    checks.append(("full-regression", ok, output[-1200:]))
    ok, output = run([sys.executable, "-m", "compileall", "-q", "waf", "tests"])
    checks.append(("compileall", ok, output[-500:]))
    ok, output = run([sys.executable, "-m", "pytest", "-q", "tests/test_phase5_explainability.py"])
    checks.append(("phase5-evidence", ok, output[-1200:]))

    passed = sum(1 for _, ok, _ in checks if ok)
    critical = [name for name, ok, _ in checks if not ok]
    score = round((passed / len(checks)) * 10.0, 2)
    report = {
        "phase": 5,
        "score": score,
        "cutoff": 9.9,
        "critical_defects": critical,
        "checks": [{"name": n, "passed": ok, "tail": text} for n, ok, text in checks],
        "evaluation_scope": "repository regression and deterministic evidence tests",
    }
    print(json.dumps(report, indent=2))
    return 0 if score >= 9.9 and not critical else 1


if __name__ == "__main__":
    raise SystemExit(main())
