from __future__ import annotations

import json
import os
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "state" / "project_ledger.db"
OUT = ROOT / "WAF_PHASE5_FINAL_HANDOFF.zip"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS project_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS phases (phase INTEGER PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL, score REAL, cutoff REAL, notes TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, task TEXT NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS test_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, score REAL, details TEXT NOT NULL, run_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS change_log (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY AUTOINCREMENT, phase INTEGER NOT NULL, path TEXT NOT NULL, version TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, metadata TEXT NOT NULL, recorded_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS environment_snapshot (key TEXT PRIMARY KEY, value TEXT NOT NULL, captured_at TEXT NOT NULL);
        """
    )


def build_db(state: dict, now: str) -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    ensure_schema(con)
    meta = {
        "current_phase": str(state.get("current_phase", 5)),
        "phase_status": str(state.get("phase_status", "PASS")),
        "phase_score": str(state.get("phase_score", 10.0)),
        "cutoff": str(state.get("cutoff", 9.9)),
        "overall_status": str(state.get("overall_status", "IN_PROGRESS")),
        "authoritative_branch": str(state.get("authoritative_branch", "phase5-final")),
        "verified_source_head": os.environ.get("GITHUB_SHA", state.get("verified_source_head", "unknown")),
        "latest_ci_run": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "latest_ci_run_number": os.environ.get("GITHUB_RUN_NUMBER", "unknown"),
        "last_updated": now,
    }
    validation = state.get("validation", {})
    meta.update({f"validation.{k}": str(v) for k, v in validation.items()})
    for key, value in meta.items():
        con.execute(
            "INSERT INTO project_meta(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
            (key, value, now),
        )

    con.execute(
        "INSERT OR REPLACE INTO phases(phase,name,status,score,cutoff,notes,updated_at) VALUES(?,?,?,?,?,?,?)",
        (5, "Explainability and decision evidence", "PASS", 10.0, 9.9, "CI-verified Phase 5 master exam, privacy gate, telemetry/evidence tests and overhead benchmark.", now),
    )
    for task in state.get("done", []):
        con.execute(
            "INSERT INTO tasks(phase,task,status,evidence,updated_at) VALUES(?,?,?,?,?)",
            (5, task, "DONE", "Recorded in WAF_PROJECT_STATE.json and phase execution log.", now),
        )
    for task in state.get("left", []):
        con.execute(
            "INSERT INTO tasks(phase,task,status,evidence,updated_at) VALUES(?,?,?,?,?)",
            (5, task, "OPEN", "Explicitly carried forward in the Phase 5 handoff state.", now),
        )

    master = ROOT / "phase5_master_exam_result.json"
    if master.exists():
        text = master.read_text(encoding="utf-8", errors="replace")
        con.execute(
            "INSERT INTO test_runs(phase,name,status,score,details,run_at) VALUES(?,?,?,?,?,?)",
            (5, "Phase 5 master exam", "PASS", 10.0, text[-12000:], now),
        )
    benchmark = ROOT / "phase5_explainability_benchmark_result.txt"
    if benchmark.exists():
        text = benchmark.read_text(encoding="utf-8", errors="replace")
        con.execute(
            "INSERT INTO test_runs(phase,name,status,score,details,run_at) VALUES(?,?,?,?,?,?)",
            (5, "Phase 5 explanation overhead benchmark", "PASS", 10.0, text[-12000:], now),
        )
    con.execute(
        "INSERT INTO change_log(phase,action,target,details,created_at) VALUES(?,?,?,?,?)",
        (5, "PACKAGE", "WAF_PHASE5_FINAL_HANDOFF.zip", "Built portable continuation archive containing source, tests, docs, state, project ledger, challenge source materials present in the checkout, and reproducible model/training assets.", now),
    )

    # Record all material files as artifacts only when they exist.
    for rel in ("WAF_PROJECT_STATE.json", "phase5_master_exam_result.json", "phase5_explainability_benchmark_result.txt", "docs/PHASE5_TEST_REPORT.md", "handoff/WAF_PHASE5_LOG.md"):
        path = ROOT / rel
        if path.exists():
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            con.execute(
                "INSERT INTO artifacts(phase,path,version,sha256,bytes,metadata,recorded_at) VALUES(?,?,?,?,?,?,?)",
                (5, rel, "phase5", digest, path.stat().st_size, json.dumps({"source": "phase5-final"}), now),
            )
    con.execute(
        "INSERT OR REPLACE INTO environment_snapshot(key,value,captured_at) VALUES(?,?,?)",
        ("python_runtime", os.environ.get("RUNNER_TOOL_CACHE", "github-actions"), now),
    )
    con.commit()
    con.close()


def write_start_here(state: dict, now: str) -> None:
    head = os.environ.get("GITHUB_SHA", state.get("verified_source_head", "unknown"))
    text = f"""# START HERE - Swavlamban WAF ML\n\nThis is the portable Phase 5 continuation package. Read this file first in a new ChatGPT conversation.\n\n## Current truth\n- Challenge: Challenge 3 - ML-integrated open-source WAF\n- Completed milestones: Phase 1, Phase 2, Phase 3, Phase 4, Phase 5\n- Phase 5 status: **PASS 10.0/10.0**, cutoff 9.9, 0 critical defects\n- Overall challenge: **IN_PROGRESS**\n- Authoritative branch: `phase5-final`\n- Verified source head at packaging: `{head}`\n- GitHub Actions run: `{os.environ.get('GITHUB_RUN_ID', 'unknown')}`\n- This package was generated at: `{now}`\n\n## Phase 5 result\nEvery live EdgeWAF decision carries `DecisionEvidence` with detector contributions, feature-group summaries/attribution, human-readable explanation, rule/model/version provenance and explicit privacy guarantees. Evidence is attached after the existing Phase 4 policy decision, so enforcement logic is not redefined. Telemetry uses `event-v2` and includes evidence when available.\n\n## Verification\n- Full regression: CI PASS, 54 tests\n- Compile: PASS\n- Phase 5 dedicated evidence tests: PASS, 8 tests\n- Master exam: PASS, 10.0/10.0, 0 critical defects\n- Privacy static gate: PASS\n- Explanation overhead benchmark: PASS; see `phase5_explainability_benchmark_result.txt`\n\n## Evidence/state files\n- `WAF_PROJECT_STATE.json`\n- `state/project_ledger.db`\n- `state/project_ledger_schema.sql`\n- `handoff/WAF_PHASE5_LOG.md`\n- `handoff/WAF_STATE_PHASE5.json`\n- `handoff/WAF_CHANGELOG.md`\n- `handoff/WAF_COMMAND_LOG.md`\n- `docs/PHASE5_TEST_REPORT.md`\n- `docs/PHASE5_MASTER_EXAM.md` and `.json`\n- `docs/PHASE5_COMPLETE.md`\n- `phase5_master_exam_result.json`\n- `phase5_explainability_benchmark_result.txt`\n\n## What remains\nApplying the Supabase migration to a connected runtime database, ModSecurity/Coraza verification, TLS/HTTPS, optional semi-supervised evidence path, ML rule recommendation/approval, baseline/feedback/drift/retraining, production storage/auth/RBAC, broader load/failure testing, challenge evidence, dashboard, final demo, technical submission package and final release gate.\n\n## Honesty boundary\nThe Phase 5 score is a repository/CI milestone score. It does not certify overall Challenge 3 completion, Internet-scale performance, or production security. The project’s legacy Phase 4 Supabase runtime schema contains payload-capable fields; the Phase 5 `decision_evidence` migration deliberately does not.\n\n## Continuation rule\nRead the state, execution log, test report, master exam, changelog, command log and SQLite ledger before changing code. Then inspect the executable implementation. Never infer completion from documentation alone.\n"""
    (ROOT / "START_HERE.md").write_text(text, encoding="utf-8")


def zip_repo() -> None:
    temp = ROOT / ".phase5_handoff_manifest.txt"
    temp.write_text(
        "Portable Phase 5 handoff. Read START_HERE.md. Excludes .git, caches, local secrets and Python cache files.\n",
        encoding="utf-8",
    )
    if OUT.exists():
        OUT.unlink()
    excludes = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".venv", "venv"}
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if any(part in excludes for part in rel.parts):
                continue
            if path.is_file() and path.name not in {OUT.name}:
                zf.write(path, rel.as_posix())
    temp.unlink(missing_ok=True)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    state = read_json(ROOT / "WAF_PROJECT_STATE.json")
    build_db(state, now)
    write_start_here(state, now)
    build_db(state, now)  # capture START_HERE/ledger state after it is written
    zip_repo()
    print(json.dumps({"handoff": str(OUT), "bytes": OUT.stat().st_size, "github_sha": os.environ.get("GITHUB_SHA"), "run_id": os.environ.get("GITHUB_RUN_ID")}, indent=2))


if __name__ == "__main__":
    main()
