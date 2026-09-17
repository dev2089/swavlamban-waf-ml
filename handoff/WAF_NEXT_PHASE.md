# Next Execution Step: Phase 8

Build production storage, authentication/RBAC, secrets handling and data-minimization hardening without weakening the Phase 5/6/7 evidence and privacy contracts.

## Phase 8 goal
Move the verified WAF control-plane state from local/in-memory demonstration components toward production-safe persistence and authorization boundaries.

## Acceptance focus
- production storage schema and migration path;
- authenticated role-aware access control;
- explicit approval permissions for rule/model promotion actions;
- secret handling and configuration hygiene;
- retention/data-minimization controls;
- regression against Phase 5 evidence, Phase 6 rule lifecycle and Phase 7 learning-control lifecycle;
- 9.9+ self-exam with critical-defect fail rule.

## Phase 7 evidence to preserve
- Baseline schema: `baseline-v1`.
- Feedback schema: `feedback-v1`.
- Drift schema: `drift-v1`.
- Model run schema: `model-run-v1`.
- Model registry schema: `model-registry-v1`.
- Human promotion is mandatory; runtime model is never silently replaced.
- Raw request material remains outside learning-control evidence.
- All Phase 7 tests, smoke evidence, registry history, ledger rows and handoff archive must remain reproducible.
