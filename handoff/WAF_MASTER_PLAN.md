# Master Execution Plan

0. Freeze + baseline evidence. DONE.
1. Architecture foundation. DONE.
2. Real HTTP interception + open-source WAF integration + actual blocking. DONE.
3. Production HTTP feature pipeline. DONE.
4. Supervised + unsupervised + behavioural ML. DONE.
5. Explainability and decision evidence.
6. Rule generation, validation, approval and deployment loop.
7. Baseline, feedback, drift and controlled retraining.
8. Production storage, auth/RBAC, secrets and data minimization.
9. Telemetry, load/performance testing and failure testing.
10. Challenge scenarios and reproducible evidence.
11. Dashboard migration to the single decision/telemetry seam.
12. Deterministic five-minute demo harness.
13. Technical documentation, slides/report and reproducibility package.
14. Final 9.9+/10 gate and release candidate.

## Phase rule
Build -> execute tests -> inspect results -> score -> remediate if <=9.8 or if any critical defect -> retest. The final project gate is blocked by any critical defect regardless of arithmetic score.

## Current milestone
Phase 4 PASS, score 10.0/10.0. The live edge now consumes signature, supervised, benign-only unsupervised and learned behavioural signals behind the `http-v2` contract.

## Phase 4 evidence
- Full regression: 48/48 PASS.
- Master exam: 10.0/10.0, 0 critical defects.
- Supervised, unsupervised and behavioural evidence are deterministic synthetic/local workloads only.
- Versioned artifact: `models/phase4_models.joblib`, schema `http-v2`, SHA-256 `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Next milestone
**Phase 5: explainability and decision evidence.**
No future milestone may silently overwrite Phase 4 evidence. Keep historical failures, remediation notes and state in the ledger.
