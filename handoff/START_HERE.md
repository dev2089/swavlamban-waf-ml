# START HERE - Swavlamban WAF ML

Read this file first in any new ChatGPT conversation.

## Current truth
- Challenge locked: **Challenge 3 - ML-integrated open-source WAF**.
- Overall project: **IN_PROGRESS**.
- Completed milestones: **Phase 1 PASS, Phase 2 PASS, Phase 3 PASS, Phase 4 PASS**.
- Current milestone: **Phase 5 implementation complete, acceptance UNVERIFIED**.
- Authoritative branch: `phase5-final`.
- Draft PR: **#1**, `Phase 5: explainability and decision evidence`.
- Main branch remains intentionally untouched by milestone work.
- Target budget: ₹0.
- Terminal is the build/test lab; GitHub is the source/control plane.
- MASTER 9.9+ quality protocol applies to every deliverable. Critical defects fail independently of arithmetic score.

## Goal
Build a real demonstrable ML-augmented WAF: live HTTP(S) inspection, open-source WAF integration, rule + ML decisions, explainability, actual allow/block enforcement, baseline/behaviour, feedback/retraining, secure telemetry/storage, dashboard, reproducible evidence and a deterministic five-minute demo.

## Phase 4 truth
- 38-feature `http-v2` input contract.
- Supervised `HistGradientBoostingClassifier`.
- Benign-only unsupervised `OneClassSVM`.
- Learned stateful behavioural `LogisticRegression`.
- Live edge consumes signature + all three ML signals.
- Artifact: `models/phase4_models.joblib`, 199123 bytes, SHA-256 `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.
- Full regression: **48/48 PASS**.
- Master exam: **10.0/10.0, 0 critical defects**.
- Latest extended local E2E: **5000 requests, 410.4 req/s, 4250 allow, 750 block, 0 errors, p50 123.614 ms, p95 160.959 ms**.
- ML metrics are deterministic synthetic benchmark/workload evidence only.

## Phase 5 truth
- Added `DecisionEvidence` schema `evidence-v1`.
- Added detector-level contributions for signature, supervised, anomaly and behaviour signals.
- Added supervised/anomaly feature-group attribution plus behavioural-state evidence.
- Added human-readable explanations and model/feature/dataset/ruleset/pipeline provenance.
- Evidence stores numeric feature values and metadata only; no raw payload, query, headers, host or source IP.
- Live `EdgeWAF` attaches evidence after the unchanged Phase 4 enforcement decision.
- Telemetry now supports evidence under `event-v2`.
- Added privacy/reproducibility tests, master exam and separate overhead benchmark.
- Added `.github/workflows/phase5.yml`.
- Added privacy-safe `decision_evidence` Supabase migration. It is committed but **not claimed applied to a live database**.
- Phase 5 runtime test score is **not claimed** because the available workflow query returned zero runs.

## Evidence locations
- `docs/PHASE5_TEST_REPORT.md`
- `handoff/WAF_PHASE5_LOG.md`
- `handoff/WAF_STATE_PHASE5.json`
- `handoff/WAF_CHANGELOG.md`
- `handoff/WAF_COMMAND_LOG.md`
- `WAF_PROJECT_STATE.json`
- `tests/test_phase5_explainability.py`
- `scripts/phase5_master_exam.py`
- `phase5_explainability_benchmark.py`
- `supabase/migrations/20260917101500_phase5_decision_evidence.sql`
- `state/project_ledger_schema.sql`
- `state/phase5_ledger.sql` when present

## Verification commands
Run these before claiming Phase 5 PASS:
- `python -m pytest -q`
- `python -m compileall -q waf tests`
- `python -m pytest -q tests/test_phase5_explainability.py`
- `python scripts/phase5_master_exam.py`
- `python phase5_explainability_benchmark.py`

## Do not regress
Do not reintroduce fake metrics, raw payloads into feature state, narrow benign baselines, shared behavioural state between edge instances, broad browser-side database scans, open public security writes, or hardcoded performance claims.

## Continuation rule
Read this file first. Then read the project state, master plan, latest phase test report/exam, execution log, changelog, command log and SQLite ledger. Inspect executable code and rerun the gates before changing project state. Never infer completion from documentation alone.

## Important honesty boundary
ModSecurity/Coraza is not claimed as separately installed/verified. TLS, optional semi-supervised work, ML rule recommendation/approval, controlled retraining/drift, production storage/auth/RBAC, full scenario evidence, dashboard, demo and final release remain open. Phase 5 implementation is complete on `phase5-final`, but acceptance remains UNVERIFIED until an actual runner produces the required 9.9+/10.0 result.
