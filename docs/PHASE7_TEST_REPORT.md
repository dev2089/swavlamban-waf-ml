# Phase 7 Test Report

## Final gate
- Score: **10.0/10.0**
- Cutoff: **9.9**
- Critical defects: **0**
- Status: **PASS**

## Checks
| Check | Result | Evidence |
|---|---|---|
| Full regression | PASS | 64/64 tests |
| Compileall | PASS | `python -m compileall -q waf tests scripts` |
| Phase 7 focused suite | PASS | 5/5 tests |
| Learning-control smoke | PASS | baseline + 44 reviewed feedback + material drift + challenger + promotion + rollback |
| Privacy/static gate | PASS | no direct raw-request field access in learning-control module |

## Deterministic smoke evidence
- Baseline version: `benign-baseline-v4-s2200-seed123-p7`
- Baseline samples: 2200
- Reviewed feedback: 44
- Drift report: `DRIFT-9D8F743279943534`
- Mean PSI: 1.064965
- Max PSI: 39.486115
- Material drift: true
- Challenger run: `RUN-CHALLENGER-267C7C9013363D97`
- Challenger model: `phase7-challenger-7cb0a4ba7530`
- Candidate evaluation: accuracy 1.0, precision 1.0, recall 1.0, F1 1.0, FPR 0.0
- Promotion: explicit human approver path exercised
- Rollback: prior `phase4-ml-v1` champion restored
- Default runtime artifact: unchanged

## Warning note
The local environment emits scikit-learn `InconsistentVersionWarning` messages because the existing model artifact was created under a different scikit-learn patch/minor version than the current local runtime. The warnings did not cause test failure. CI is the independent release-verification environment and remains the authoritative external gate for the branch.

## Scope boundary
These results establish deterministic implementation correctness and reproducibility. They are not a claim of production-scale accuracy or a live production deployment validation.
