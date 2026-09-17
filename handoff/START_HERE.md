# WAF Future-Chat START HERE

## Current control point
Phase 10 is the release-candidate milestone. Phase 1-9 evidence is preserved. The authoritative Phase 10 candidate must be the exact `phase10-final` branch tip after its final green CI run.

## Read first
1. `WAF_PROJECT_STATE.json`
2. `handoff/PHASE10_FINAL_STATUS.md`
3. `handoff/WAF_PHASE10_LOG.md`
4. `phase10_master_exam_result.json`
5. `handoff/PHASE10_REQUIREMENT_TRACEABILITY.json`
6. `handoff/PHASE10_PRODUCTION_READINESS.json`
7. `handoff/PHASE10_CLAIM_LEDGER.json`
8. `handoff/PHASE10_NEGATIVE_EVIDENCE.md`
9. `handoff/PHASE10_SUPABASE_LIVE_VERIFICATION.json`
10. `state/phase10_ledger.sql`
11. Phase 9 status/log and earlier historical milestone evidence

## Final runtime/release controls
- `waf/gateway/proxy.py`
- `waf/edge/pipeline.py`
- `waf/ml/ensemble.py`
- `waf/ml/semisupervised.py`
- `waf/ml/behaviour.py`
- `waf/ml/outbound.py`
- `deploy/nginx/phase10-modsecurity.conf`
- `scripts/phase10_master_exam.py`
- `scripts/phase10_waf_enforcement_e2e.py`
- `scripts/phase10_tls_e2e.py`
- `scripts/phase10_outbound_e2e.py`
- `scripts/phase10_rule_replay.py`
- `scripts/phase10_load_harness.py`
- `scripts/phase10_dashboard_demo.py`
- `.github/workflows/phase10.yml`
- `dashboard/index.html`

## ML contract
Request feature schema: `http-v2`.
Request detectors: supervised, unsupervised, semi-supervised, behavioural.
Outbound response schema: `http-response-v1`.
Outbound detector: `outbound-oneclasssvm-v1`.
Model artifact schema: `phase10-model-v3`.

## Live Supabase
Project: `smpmvabjafmrutdhbfbl` (`supabase-pink-village`). Live schema/RLS/privilege verification and security hardening evidence are preserved in the Phase 10 package. Secrets are not stored in the repository or handoff.

## Reproduction
```bash
python -m pytest -q
python -m compileall -q waf tests scripts
python scripts/phase10_master_exam.py
```
The CI gate additionally builds binary submission artifacts, captures dependency inventory and exact commit, builds the portable auditor handoff, records checksums and uploads the transfer artifacts.

## Evidence discipline
Measured evidence, deterministic synthetic evaluation, design projections and external constraints are separate categories. Never treat historical milestone prose as current runtime proof. Never treat a README, mock, migration or generated label as evidence unless an executable check connects it to behavior.

## Remaining external boundaries
Public certificate issuance/rotation, public Internet HTTPS verification, Internet-scale distributed capacity, venue-specific public deployment and final challenge portal operations remain outside local CI proof. They are explicit non-claims, not hidden omissions.

## Continuation rule
Preserve all prior failures/remediations and the exact release branch history. Human approval and no-auto-promotion controls remain mandatory. Raw payload/query/header material must not be added to persisted telemetry or audit bundles.
