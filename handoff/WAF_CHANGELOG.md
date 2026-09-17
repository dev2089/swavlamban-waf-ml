# WAF Change Log

## Cycle 0
- Challenge 3 locked after documented comparison.
- Baseline main commit recorded: `1cc4f91dd6828039f834ae4dc2b466191d04f229`.
- Prototype/production gap and major ML, WAF, security and documentation mismatches recorded.

## Phase 1
- Added dependency-light canonical security core and tests.
- Added architecture/state/handoff documentation and portable SQLite ledger.
- Gate: 10.0/10.0.

## Phase 2
- Added live HTTP reverse-proxy enforcement and Nginx integration.
- Added deterministic URL-decoded SQLi/XSS/traversal/command signatures.
- Added actual pre-forwarding HTTP 403 enforcement, bounds, timeout handling and WAF headers.
- Gate: 10.0/10.0 for Phase 2 acceptance criteria.

## Phase 3
- Added `http-v2` with 38 bounded normalized HTTP features.
- Added safe decoding, Unicode normalization and malformed-input handling.
- Added feature fuzz/performance/live-edge evidence and regression coverage.
- Gate: 10.0/10.0 for Phase 3 acceptance criteria.

## Phase 4
- Added supervised `HistGradientBoostingClassifier`.
- Added benign-only `OneClassSVM` anomaly detection with learned benign threshold.
- Rebuilt behavioural detector as learned `LogisticRegression` over stateful burst/churn features.
- Expanded benign training baseline variation to reduce false positives.
- Persisted all three model components in a versioned artifact.
- Added artifact completeness validation, immutable-model caching and fresh per-edge behavioural state.
- Added supervised/unsupervised/behavioural evaluation, artifact round-trip, live-edge and master-exam gates.
- Added extended 5,000-request E2E evidence.
- Added artifact/environment tables to the portable project ledger.
- Preserved all failed cycles and remediation decisions in the phase log and test ledger.
- Final Phase 4 gate: 10.0/10.0 with 0 critical defects.

## Final verification refresh
- Regenerated the authoritative Phase 4 model manifest after learned-behaviour hardening.
- Re-ran full regression: 48/48 PASS.
- Re-ran master exam: 10.0/10.0, critical defects 0.
- Latest mandatory direct ML benchmark: 2,000 requests, 531.1 req/s, 1,900 allow, 100 block.
- Latest bounded E2E: 1,000 requests, 428.6 req/s, 850 HTTP 200, 150 HTTP 403, 0 errors.
- Latest extended E2E: 5,000 requests, 410.4 req/s, 4,250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms.
- Handoff smoke test from the final ZIP: 48/48 regression PASS, compileall PASS, master exam PASS and self-test PASS.

## Phase 5
- Added `DecisionEvidence` (`evidence-v1`) to the core result model without breaking legacy results.
- Added deterministic detector-level risk contribution accounting.
- Added supervised/anomaly feature-group attribution and behavioural-state evidence.
- Added human-readable explanations and model/feature/dataset/ruleset/pipeline provenance.
- Added privacy guarantees and leakage tests; evidence contains numeric feature snapshots but no raw payload/query/header/source-IP material.
- Wired evidence into live `EdgeWAF` after Phase 4 policy enforcement.
- Extended telemetry to `event-v2` with optional evidence.
- Added Phase 5 tests, master exam, overhead benchmark and CI workflow.
- Added privacy-safe `decision_evidence` Supabase migration; the migration is committed but not claimed applied to a live database.
- Created draft PR #1 from `phase5-final` to `main`; main remains untouched.
- GitHub Actions query returned zero workflow runs for the Phase 5 head, so no Phase 5 passing score is claimed.

## Important boundary
- Main branch remains untouched by milestone work.
- Synthetic metrics are not real-world accuracy claims.
- Phase 5 implementation is complete on `phase5-final`, but acceptance verification is UNVERIFIED until a real runner executes the gates.
- ModSecurity/Coraza verification, TLS, optional semi-supervised evidence, rule lifecycle, controlled retraining, production storage/auth, dashboard, final demo and release remain open.
- Overall Challenge 3 remains IN_PROGRESS.
