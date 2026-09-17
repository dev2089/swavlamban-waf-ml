"""Hard-gated Phase 10 release-candidate exam."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(label: str, command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {"label": label, "ok": completed.returncode == 0, "returncode": completed.returncode, "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}

def main() -> int:
    checks = [
        run("full_regression", [sys.executable, "-m", "pytest", "-q"]),
        run("compile", [sys.executable, "-m", "compileall", "-q", "waf", "tests", "scripts"]),
        run("phase10_demo", [sys.executable, "scripts/phase10_demo.py"]),
    ]
    required = [
        "dashboard/index.html",
        "docs/PHASE10_TECHNICAL_REPORT.md",
        "docs/PHASE10_PRESENTATION.md",
        "scripts/phase10_demo.py",
        "scripts/phase10_master_exam.py",
        "handoff/WAF_PHASE9_LOG.md",
        "handoff/PHASE9_FINAL_STATUS.md",
        "WAF_PROJECT_STATE.json",
    ]
    checks.append({"label": "required_artifacts", "ok": all((ROOT / p).exists() for p in required), "missing": [p for p in required if not (ROOT / p).exists()]})
    checks.append({"label": "evidence_boundary", "ok": "Internet-scale" in (ROOT / "docs/PHASE10_TECHNICAL_REPORT.md").read_text(encoding="utf-8") and "ModSecurity/Coraza" in (ROOT / "docs/PHASE10_TECHNICAL_REPORT.md").read_text(encoding="utf-8")})
    passed = sum(bool(x["ok"]) for x in checks)
    score = 10.0 if all(bool(x["ok"]) for x in checks) else max(0.0, round(10.0 - (len(checks) - passed), 2))
    result = {"phase": 10, "score": score, "cutoff": 9.9, "critical_defects": 0 if score >= 9.9 else 1, "checks": checks, "passed": passed, "total": len(checks)}
    (ROOT / "phase10_master_exam_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if score >= 9.9 and all(bool(x["ok"]) for x in checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
