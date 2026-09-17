# Swavlamban WAF ML

ML-integrated open-source WAF for Challenge 3.

## Current milestone
**Phase 9 complete under the 9.9 hard gate:** 10.0/10.0, 82/82 full regression PASS, 13/13 Phase 8 continuity PASS, 5/5 Phase 9 API tests PASS, compile/static/TLS/scenario gates PASS, zero critical failures.

Authoritative branch: `phase9-final`.

Phase 9 also verified the live Supabase project `supabase-pink-village` (`smpmvabjafmrutdhbfbl`): the WAF schema and security migration chain is applied, expected tables exist, RLS is enabled, anonymous runtime access is closed, and runtime writes are service-role controlled.

## Read-first continuation
See `WAF_PROJECT_STATE.json` and `handoff/START_HERE.md`. Then read `handoff/PHASE9_FINAL_STATUS.md`, `handoff/WAF_PHASE9_LOG.md`, `phase9_master_exam_result.json`, `phase9_scenario_evidence.json`, and the portable ledger under `state/`.

## Phase 9 completed
- secure production-facing FastAPI runtime adapter
- signed bearer authentication and explicit RBAC
- server-only Supabase REST persistence
- privacy-safe telemetry and SHA-256 source identity hashing
- legacy runtime-table lockdown and raw-request cleanup
- local nginx HTTPS/TLS integration
- deterministic Challenge 3 attack/scenario evidence
- 500-request in-process performance measurement
- executable master exam, ledger, remediation history and future-chat handoff
- live Supabase schema/RLS/privilege verification

## Still open for final submission
External certificate/public HTTPS operations, ModSecurity/Coraza runtime integration, Internet-scale distributed load/failure validation, authenticated dashboard UX, five-minute demo, technical report, presentation slides and final release/submission gate. GitHub Actions Phase 9 CI remains `NOT_OBSERVED` until a real PR-triggered run is returned.
