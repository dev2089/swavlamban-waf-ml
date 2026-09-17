# Swavlamban WAF ML - START HERE

## Current control point
Phase 9 is complete under the hard 9.9 gate: **10.0/10.0**, **82/82 full regression PASS**, **13/13 Phase 8 continuity PASS**, **5/5 Phase 9 API tests PASS**, compile/static/TLS/scenario gates PASS, zero critical failures.

Authoritative branch: `phase9-final`.

## Read first for any future ChatGPT conversation
1. `WAF_PROJECT_STATE.json`
2. `handoff/START_HERE.md`
3. `handoff/PHASE9_FINAL_STATUS.md`
4. `handoff/WAF_PHASE9_LOG.md`
5. Phase 9 scenario/master-exam evidence and the portable `state/` ledger
6. Phase 8 status/log and earlier milestone records

## Live database checkpoint
The connected Supabase project is `smpmvabjafmrutdhbfbl` (`supabase-pink-village`), currently `ACTIVE_HEALTHY`. Phase 9 applied and verified the WAF schema/security migration chain, RLS and privilege lockdown. Anonymous runtime access is closed and runtime persistence writes are server/service-role controlled.

## Important boundaries
Phase 9 does not claim external certificate issuance/rotation, public HTTPS verification, ModSecurity/Coraza installation, Internet-scale distributed load/failure validation, authenticated dashboard UX, demo recording, report/slides, or final submission gate. GitHub Actions Phase 9 CI is not claimed until a real PR-triggered run is observed.

## Continuation rule
Preserve all Phase 1-9 evidence and every failure/remediation record. For non-GitHub project connectors, start the operation context with `swavlamban-waf`.
