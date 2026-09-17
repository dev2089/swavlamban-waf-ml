# Swavlamban WAF ML

ML-augmented Web Application Firewall prototype for **Challenge 3**.

## Verified milestones
- Phase 1: PASS, 10.0/10.0
- Phase 2: PASS, 10.0/10.0
- Phase 3: PASS, 10.0/10.0
- Phase 4: PASS, 10.0/10.0, critical defects 0
- Phase 5: PASS, 10.0/10.0, 54/54 regression
- Phase 6: PASS, 10.0/10.0, 59/59 regression
- Phase 7: PASS, 10.0/10.0, 64/64 regression
- Phase 8: PASS locally, 10.0/10.0, 70/70 regression, 6/6 security tests, 0 critical defects
- Phase 9: PASS locally, 10.0/10.0, 82/82 regression, 13/13 Phase 8 continuity, 5/5 Phase 9 API tests, 0 failed critical checks

## Phase 9
Production-facing FastAPI integration now enforces the Phase 8 signed bearer-token/RBAC model at runtime. Runtime persistence has a server-only Supabase REST adapter, sanitized telemetry and SHA-256 source identity hashing. A Phase 9 SQL migration locks legacy runtime tables behind authenticated roles, removes permissive anonymous access and scrubs raw request material.

Controlled evidence includes local nginx TLS termination, known SQL/XSS and command-variant scenarios, API-burst behavioural detection, and a 500-request in-process performance benchmark. The executable Phase 9 master exam applies a hard 9.9 cutoff and zero-critical-failure rule.

Run the Phase 9 gate:
```bash
python -m pytest -q
python -m compileall -q waf tests scripts
python scripts/phase9_master_exam.py
python scripts/record_phase9_ledger.py
python scripts/build_phase9_handoff.py
```

## Evidence and continuation
Read `handoff/START_HERE.md` first. The portable project ledger is `state/project_ledger.db`; the Phase 9 SQL checkpoint is `state/phase9_ledger.sql`; executable evidence is in `phase9_master_exam_result.json`, `phase9_scenario_evidence.json` and `phase9_tls_evidence.json`.

## Honesty boundary
Phase 9 is a repository/local integration milestone. It does not claim live Supabase migration/application, external public certificate operations, ModSecurity/Coraza installation, Internet-scale load/failure testing, authenticated dashboard completion, final demo/report/slides or final Challenge 3 submission. Local TLS and performance measurements are not external-deployment proof.
