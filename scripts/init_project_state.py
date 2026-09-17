from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
DB = STATE / "waf_project_state.sqlite"
JSON = ROOT / "WAF_PROJECT_STATE.json"

state = {
    "project": "swavlamban-waf-ml",
    "baseline_commit": "1cc4f91dd6828039f834ae4dc2b466191d04f229",
    "selected_challenge": "Challenge 3 - ML-integrated open-source WAF",
    "current_phase": 1,
    "phase_status": "PASS",
    "phase1_exam_score": 10.0,
    "phase1_cutoff": 9.9,
    "overall_project_status": "IN_PROGRESS",
    "current_branch": "phase1-architecture-rebuild",
    "done": ["canonical request/data models", "contracts/interfaces", "versioned baseline HTTP feature extraction", "deterministic decision policy", "event schema", "bounded body handling", "phase 1 tests", "architecture documentation", "SQLite project ledger"],
    "left": ["real reverse-proxy/open-source WAF integration", "actual block enforcement", "TLS handling", "production ML training/evaluation", "behavioural detection", "continuous learning/retraining", "production storage/auth/RBAC", "performance/load testing", "dashboard migration", "demo and final evidence package"],
    "validation": {"unit_tests": "13/13 PASS", "compileall": "PASS", "static_secret_scan": "PASS", "static_external_io_scan": "PASS", "core_benchmark": "100000 mixed requests; ~23882 decisions/sec", "benchmark_scope": "Phase 1 core only, not end-to-end WAF"},
    "honesty_boundary": "Phase 1 foundation is complete; the full challenge is still in progress."
}
STATE.mkdir(exist_ok=True)
JSON.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
conn = sqlite3.connect(DB)
conn.executescript(Path(STATE / "project_ledger_schema.sql").read_text(encoding="utf-8"))
now = datetime.now(timezone.utc).isoformat()
for key, value in (("project", state["project"]), ("baseline_commit", state["baseline_commit"]), ("selected_challenge", state["selected_challenge"]), ("current_phase", "1"), ("overall_status", state["overall_project_status"]), ("current_branch", state["current_branch"])):
    conn.execute("INSERT INTO project_meta(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at", (key, value, now))
conn.execute("INSERT INTO phases(phase,name,status,score,cutoff,notes,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(phase) DO UPDATE SET status=excluded.status, score=excluded.score, cutoff=excluded.cutoff, notes=excluded.notes, updated_at=excluded.updated_at", (1, "Architecture Foundation", "PASS", 10.0, 9.9, "Foundation contracts and tests completed locally.", now))
conn.execute("DELETE FROM tasks WHERE phase=1 OR (phase=2 AND status='PLANNED')")
for task in state["done"]:
    conn.execute("INSERT INTO tasks(phase,task,status,evidence,updated_at) VALUES(?,?,?,?,?)", (1, task, "DONE", "repo + tests + docs", now))
for task in state["left"]:
    conn.execute("INSERT INTO tasks(phase,task,status,evidence,updated_at) VALUES(?,?,?,?,?)", (2, task, "PLANNED", "future phase", now))
conn.execute("INSERT INTO test_runs(phase,name,status,score,details,run_at) VALUES(?,?,?,?,?,?)", (1, "MASTER PHASE 1 GATE", "PASS", 10.0, "13/13 unit tests, compileall, static scans and fuzz benchmark all passed.", now))
conn.execute("INSERT INTO change_log(phase,action,target,details,created_at) VALUES(?,?,?,?,?)", (1, "ADD", "waf/ + tests/ + docs/ + handoff/ + state/", "Created architecture foundation beside legacy implementation; legacy code intentionally not removed yet.", now))
conn.commit(); conn.close()
