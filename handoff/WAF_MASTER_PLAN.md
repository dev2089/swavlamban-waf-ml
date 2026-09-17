# Master Execution Plan

0. Freeze + baseline evidence. DONE.
1. Architecture foundation. DONE.
2. Real HTTP interception + open-source WAF integration + actual blocking. DONE.
3. Production HTTP feature pipeline. DONE.
4. Supervised + unsupervised + behavioural ML. DONE.
5. Explainability and decision evidence. DONE.
6. Rule generation, validation, approval and deployment loop. DONE.
7. **Baseline, feedback, drift and controlled retraining. DONE / VERIFIED.**
8. Production storage, auth/RBAC, secrets and data minimization. NEXT MILESTONE.
9. Telemetry, load/performance testing and failure testing. OPEN.
10. Challenge scenarios and reproducible evidence. OPEN.
11. Dashboard migration to the single decision/telemetry seam. OPEN.
12. Deterministic five-minute demo harness. OPEN.
13. Technical documentation, slides/report and reproducibility package. OPEN.
14. Final 9.9+/10 gate and release candidate. OPEN.

## Phase rule
Build -> execute tests -> inspect results -> score -> remediate if <=9.8 or if any critical defect -> retest. The final project gate is blocked by any critical defect regardless of arithmetic score.

## Phase 7 definition
Phase 7 adds a reproducible learning-control loop over the existing `http-v2` feature contract: a versioned benign baseline, human-reviewed feedback, deterministic drift detection, controlled challenger retraining, frozen champion-versus-challenger evaluation, explicit promotion, and explicit rollback. The known-good runtime artifact is never silently replaced and raw request material is excluded from learning-control records.

## Historical continuity
Phase 5 remains preserved in `phase5-final`, Phase 6 remains preserved in `phase6-final`, and Phase 7 is authoritative on `phase7-final`. Earlier milestone evidence must remain intact and readable from the portable handoff.

## Phase 7 verification checkpoint
Phase 7 local verification: 64/64 regression PASS, 5/5 focused tests PASS, compileall PASS, deterministic learning-control smoke PASS, privacy/static gate PASS, master exam 10.0/10.0 with cutoff 9.9 and zero critical defects. CI verification is recorded in the Phase 7 state/log when the authoritative workflow completes.

## Next milestone after Phase 7
**Phase 8: production storage, authentication/RBAC, secrets and data-minimization hardening.**
