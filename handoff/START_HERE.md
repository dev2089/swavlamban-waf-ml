# START HERE - Swavlamban WAF ML

Read this file first in any new ChatGPT conversation.

## Current truth
- Challenge locked: **Challenge 3 - ML-integrated open-source WAF**.
- Overall project: **IN_PROGRESS**.
- Completed milestones: **Phase 1 PASS, Phase 2 PASS, Phase 3 PASS**.
- Authoritative branch: `phase3-final`.
- Main branch remains intentionally untouched by milestone work.
- Target budget: ₹0.
- Terminal workspace is the build/test lab; GitHub is the source/control plane.
- MASTER 9.9+ quality protocol applies to all deliverables. Critical defects fail independently of arithmetic score.

## Goal
Build a real demonstrable ML-augmented WAF: live HTTP(S) inspection, open-source WAF integration, rule + ML decisions, explainability, actual allow/block enforcement, baseline/behaviour, feedback/retraining, secure telemetry/storage, dashboard, reproducible evidence and a deterministic five-minute demo.

## Milestone state
- Phase 1: architecture foundation, 10.0/10.0.
- Phase 2: live HTTP interception + Nginx + actual block enforcement, 10.0/10.0.
- Phase 3: production HTTP feature pipeline, 10.0/10.0.

## Phase 3 specifics
- Feature schema: `http-v2`.
- Feature count: 38.
- Path/query-safe multi-pass URL decoding.
- Unicode NFKC normalization.
- Query/header/body bounds.
- Numeric output only, values in `[0,1]`.
- Edge WAF is wired to v2.
- Local full regression: 20/20 PASS.
- Feature fuzz: 20,000 requests, 0 exceptions.
- Feature benchmark: 100,000 extractions, 34,854.5 req/s.
- E2E benchmark: 5,000 requests, 4,500 allowed, 500 blocked, 2,354.3 req/s.

## Next milestone
Phase 4: supervised + unsupervised + behavioural ML using the same canonical request/feature contracts.

## Do not regress
Do not reintroduce raw payloads into feature state, fake model metrics, broad browser-side database scans, open public security writes, or hardcoded performance claims.

## Evidence locations
- `docs/PHASE3_COMPLETE.md`
- `docs/PHASE3_TEST_REPORT.md`
- `docs/HTTP_FEATURE_SCHEMA_V2.md`
- `handoff/WAF_PHASE3_LOG.md`
- `WAF_PROJECT_STATE.json`
- `handoff/WAF_STATE_PHASE3.json`
- `handoff/WAF_CHANGELOG.md`
- `handoff/WAF_COMMAND_LOG.md`
- `state/project_ledger.db` in the local handoff ZIP

## Important honesty rule
Never infer completion from documentation alone. Read executable code and run the tests before changing state.
