# WAF Handoff Context

## Project
- Repository: `dev2089/swavlamban-waf-ml`
- Challenge: **Challenge 3 - ML-integrated open-source WAF**
- Baseline main commit inherited at project start: `1cc4f91dd6828039f834ae4dc2b466191d04f229`
- Authoritative release branch: `phase10-final`
- Current engineering candidate is maintained with complete Phase 1-10 historical continuity.

## Quality rule
The internal release gate is 100% on applicable executable acceptance checks, with any critical defect causing failure. Documentation alone never counts as proof. Historical evidence is preserved even when later phases replace or strengthen an implementation.

## Current Phase 10 scope
The release candidate now contains:
- real HTTP interception through the Swavlamban gateway;
- nginx + ModSecurity open-source WAF enforcement before a protected upstream;
- request features under `http-v2`;
- supervised, unsupervised, semi-supervised and stateful behavioural request detectors;
- outbound HTTP response anomaly inspection under `http-response-v1`;
- structured decision explainability and version provenance;
- rule generation, replay validation, human approval, deployment and rollback;
- baseline/feedback/drift/challenger retraining controls from Phase 7;
- authenticated FastAPI control plane and role-based administrative actions;
- bounded asynchronous/server-side telemetry with raw request material excluded;
- live Supabase schema/RLS/privilege verification;
- authenticated dynamic dashboard;
- deterministic TLS termination, API/bot/zero-day-style evidence, load harness and five-minute browser demo;
- technical report, presentation source, audit/claim/negative-evidence manifests and portable handoff.

## ML runtime contract
Model artifact schema: `phase10-model-v3`.
Request feature schema: `http-v2`.
Outbound response feature schema: `http-response-v1`.
Request detectors: `supervised-v1`, `unsupervised-oneclasssvm-v1`, `behaviour-v1`, `semi-supervised-v1`.
Outbound detector: `outbound-oneclasssvm-v1`.

The project uses deterministic versioned synthetic datasets for reproducible local evaluation. Those metrics are not claims of field or Internet-wide WAF accuracy.

## Verification
The final candidate must be verified from a clean checkout. The authoritative CI gate runs compile, full regression, real gateway/WAF/TLS/replay/outbound tests, bounded network load, the dashboard demo, repository hygiene, audit generation and submission artifact packaging.

## Explicit boundaries
Public certificate issuance/rotation and public Internet HTTPS verification are external to the local proof. Internet-scale distributed capacity is not physically demonstrated in the free environment. Venue-specific public deployment and final challenge upload remain operational steps rather than source-code claims.

## Continuation rule
A future ChatGPT conversation should start from the exact branch tip and read `WAF_PROJECT_STATE.json`, `handoff/START_HERE.md`, `handoff/PHASE10_FINAL_STATUS.md`, `handoff/WAF_PHASE10_LOG.md`, `handoff/PHASE10_REQUIREMENT_TRACEABILITY.json`, `handoff/PHASE10_PRODUCTION_READINESS.json`, `handoff/PHASE10_CLAIM_LEDGER.json`, `handoff/PHASE10_NEGATIVE_EVIDENCE.md`, the final master-exam result and the complete source archive before changing anything.
