# Phase 5 Test Report

## Scope
Phase 5 adds structured explainability and decision evidence to the Phase 4 live WAF path. The report deliberately distinguishes executable test definitions from executed results.

## Required gates
| Gate | Command | Required result | Current recorded result |
|---|---|---|---|
| Full regression | `python -m pytest -q` | PASS | UNVERIFIED in this connector session |
| Compile | `python -m compileall -q waf tests` | PASS | UNVERIFIED in this connector session |
| Phase 5 evidence tests | `python -m pytest -q tests/test_phase5_explainability.py` | PASS | UNVERIFIED in this connector session |
| Master exam | `python scripts/phase5_master_exam.py` | >= 9.9/10.0, zero critical defects | UNVERIFIED in this connector session |
| Explanation overhead | `python phase5_explainability_benchmark.py` | separate reproducible measurement | UNVERIFIED in this connector session |

## What is statically evidenced
- `DecisionEvidence` is versioned as `evidence-v1`.
- `DecisionResult.evidence` is optional, so existing Phase 4 result construction remains valid.
- Live `EdgeWAF` performs the same signature/ML/policy decision and then attaches evidence.
- Evidence privacy fields hard-fail construction if raw payload/query/header retention is declared true.
- Evidence contains numeric feature snapshots and feature-group metadata rather than raw request material.
- Telemetry emits evidence only when it exists.
- Dedicated tests cover benign, known signature attack, unseen anomaly, deterministic evidence and legacy compatibility.
- A Phase 5 master exam and CI workflow are present.

## Phase 4 regression baseline
Before Phase 5, the repository recorded 48/48 tests passing and a 10.0/10.0 Phase 4 master exam with zero critical defects. Phase 5 preserves that baseline as the regression target rather than silently replacing it.

## Verification boundary
No Phase 5 numeric score is claimed here because the available GitHub Actions query returned no workflow run for the Phase 5 head commit. This report therefore does not manufacture a passing result. A future runner must execute the commands above and append the actual outputs.
