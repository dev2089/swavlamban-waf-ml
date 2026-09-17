# Swavlamban WAF ML

ML-augmented Web Application Firewall prototype for **Challenge 3**.

## Verified milestones
- Phase 1: PASS, 10.0/10.0
- Phase 2: PASS, 10.0/10.0
- Phase 3: PASS, 10.0/10.0
- Phase 4: PASS, 10.0/10.0, critical defects 0
- Phase 7: PASS, 10.0/10.0, 64/64 regression
- Phase 8: PASS, 10.0/10.0, 70/70 regression, 6/6 security tests, 0 critical defects

## Phase 8
Production-security control contracts now include signed bearer authentication, explicit RBAC, fail-closed configuration validation, privacy-safe audit records/security headers, and a Supabase RLS migration path.

Run the Phase 8 gate:
```bash
python -m pytest -q
python -m compileall -q waf tests scripts
python scripts/phase8_master_exam.py
```

## Evidence and continuation
Read `handoff/START_HERE.md` first. The portable project ledger is `state/project_ledger.db`; the Phase 8 textual checkpoint is `state/phase8_ledger.sql`.

## Honesty boundary
Phase 8 is a repository-level security milestone. It does not claim live Supabase migration/application deployment, production FastAPI endpoint auth wiring, external TLS/HTTPS verification, ModSecurity/Coraza verification, complete external load/failure evidence, or final Challenge 3 completion. Synthetic ML metrics remain reproducibility evidence only.
