# WAF Phase 7 Execution Log

## Objective
Build the reproducible learning-control loop from the Phase 6 handoff: benign baseline, reviewed feedback, material drift detection, controlled retraining, challenger-versus-champion evaluation, explicit promotion and rollback.

## Implementation completed
`waf/ml/learning_control.py` adds the Phase 7 contracts and lifecycle. The live edge path now records Phase 7 model/baseline provenance in Phase 5 decision evidence without persisting raw request material.

## Acceptance path executed
1. Generated the versioned benign baseline from `http-v2` numeric features.
2. Created 44 deterministic feedback records and explicitly reviewed every label.
3. Generated a deterministic material-drift workload and received a material drift alert.
4. Trained a challenger from the existing deterministic corpus plus reviewed feedback.
5. Evaluated challenger and champion on the same frozen synthetic validation benchmark.
6. Exercised explicit human promotion.
7. Exercised explicit rollback to the prior champion.
8. Verified the default `models/phase4_models.joblib` artifact was not silently replaced.
9. Ran full regression, compile, focused tests and privacy/static gates.

## Final verification
- Master exam: PASS 10.0/10.0.
- Cutoff: 9.9.
- Critical defects: 0.
- Full regression: 64/64 PASS.
- Focused Phase 7 tests: 5/5 PASS.
- Drift smoke: mean PSI 1.064965, max PSI 39.486115, material=true.
- Candidate: `RUN-CHALLENGER-267C7C9013363D97`.
- Candidate model: `phase7-challenger-7cb0a4ba7530`.
- Candidate evaluation: accuracy/precision/recall/F1 1.0, FPR 0.0 on deterministic synthetic validation.
- Promotion and rollback: PASS.

## Evidence/privacy boundary
Baseline and feedback records contain only bounded numeric `http-v2` features plus provenance metadata. The learning-control module has no direct access to raw body/query/header/host/source-IP fields. Promotion is never automatic.

## Database
The SQLite project ledger was extended with Phase 7 baseline, feedback, drift, model-run, promotion and rollback tables. `state/phase7_ledger.sql` is the portable textual checkpoint. The actual SQLite database is included in the local final handoff archive.

## Open work after Phase 7
Production storage/auth/RBAC/secrets, live Supabase application, ModSecurity/Coraza verification, TLS deployment verification, load/failure testing, challenge evidence, dashboard/demo/report and the final release gate remain open. These are carried forward deliberately rather than hidden behind the Phase 7 score.
