#!/usr/bin/env python3
"""Build a portable Phase 7 continuation archive from the verified checkout."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "WAF_PHASE7_FINAL_HANDOFF.zip"
EXCLUDE = {".git", ".pytest_cache", "__pycache__", "node_modules"}


def files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDE for part in rel.parts):
            continue
        if rel.name in {"WAF_PHASE7_FINAL_HANDOFF.zip", "WAF_PHASE6_FINAL_HANDOFF.zip"}:
            continue
        yield path, rel


def main() -> None:
    result = json.loads((ROOT / "phase7_master_exam_result.json").read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or float(result.get("score", 0)) < 9.9 or result.get("critical_defects"):
        raise SystemExit("Phase 7 handoff refused: master exam is not a clean PASS")
    required = [
        "waf/ml/learning_control.py", "tests/test_phase7_learning_control.py", "scripts/phase7_master_exam.py",
        "docs/PHASE7_COMPLETE.md", "handoff/PHASE7_FINAL_STATUS.json", "handoff/WAF_PHASE7_LOG.md", "handoff/START_HERE.md",
    ]
    if not (ROOT / "state/project_ledger.db").exists() or not (ROOT / "state/phase7_ledger.sql").exists():
        raise SystemExit("Phase 7 handoff missing generated project ledger; run scripts/record_phase7_ledger.py first")
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Phase 7 handoff missing required evidence: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, rel in sorted(files(), key=lambda item: str(item[1])):
            archive.write(path, rel.as_posix())
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(json.dumps({"archive": OUT.name, "bytes": OUT.stat().st_size, "sha256": sha, "files": sum(1 for _ in files()), "phase7_score": result["score"], "critical_defects": result["critical_defects"], "ledger_included": True}, indent=2))


if __name__ == "__main__":
    main()
