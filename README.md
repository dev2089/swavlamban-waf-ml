# Swavlamban WAF ML

ML-integrated open-source WAF for **Challenge 3**.

## Current milestone
**Phase 10 release candidate complete at the 9.9 hard gate:** 10.0/10.0, 78/78 locally runnable regression, compile PASS, dashboard/release tests PASS, deterministic demo PASS, zero critical defects. The authoritative Phase 9 record remains preserved at 82/82 regression PASS.

Authoritative branch: `phase10-final`.

Live Supabase project `supabase-pink-village` (`smpmvabjafmrutdhbfbl`) is `ACTIVE_HEALTHY`. The WAF baseline through Phase 10 database-control migrations is applied and verified; RLS is enabled across the tracked WAF control/runtime tables. The Supabase security advisor is clear after function-search-path hardening. Performance advice is INFO-only for fresh-schema unused indexes.

## Read-first continuation
Read `WAF_PROJECT_STATE.json` and `handoff/START_HERE.md`, then `handoff/PHASE10_FINAL_STATUS.md`, `handoff/WAF_PHASE10_LOG.md`, `phase10_master_exam_result.json`, `phase10_demo_evidence.json`, `handoff/PHASE10_SUPABASE_LIVE_VERIFICATION.json`, and `state/phase10_ledger.sql`.

## Phase 10 release candidate
- authenticated operator dashboard at `/dashboard`
- secure `/api/release` evidence endpoint
- deterministic benign/SQL/XSS/command-variant scenario runner
- 500-request in-process benchmark with explicit boundary
- hard-gated release exam
- technical report and ten-slide presentation source
- live Supabase release audit plus security/performance remediation migrations
- complete Phase 1-10 evidence, remediation history and future-chat handoff

## Reproduce
```bash
python -m pytest -q
python -m compileall -q waf tests scripts
python scripts/phase10_demo.py
python scripts/phase10_master_exam.py
```

## Evidence boundary
The repository does **not** claim public certificate issuance/rotation, public Internet HTTPS verification, ModSecurity/Coraza installation, or Internet-scale distributed load/failure validation. Those remain explicitly unverified/open until independently evidenced.

Final venue-specific work may still include the five-minute recording, binary slide export if required, and final submission upload/checklist.
