# Phase 5 Test Report

## Scope
Phase 5 adds structured explainability and decision evidence to the Phase 4 live WAF path. The report distinguishes implementation evidence from executed verification and preserves the limits of benchmark interpretation.

## Final executed verification
| Gate | Command | Final result | Evidence |
|---|---|---|---|
| Full regression | `python -m pytest -q` | **54/54 PASS** | GitHub Actions run `35210837101` and later documentation-triggered run `35211601235` both completed successfully |
| Compile | `python -m compileall -q waf tests` | **PASS** | GitHub Actions |
| Phase 5 evidence tests | `python -m pytest -q tests/test_phase5_explainability.py` | **8/8 PASS** | Phase 5 master exam gate |
| Master exam | `python scripts/phase5_master_exam.py` | **10.0/10.0**, cutoff 9.9, critical defects 0 | GitHub Actions run `35210837101` |
| Privacy static gate | included in master exam | **PASS** | Raw payload/query/header/source-IP storage checks passed |
| Explanation overhead | `python phase5_explainability_benchmark.py` | **PASS** | Local deterministic workload benchmark, timing-dependent |

## Phase 5 acceptance result
Phase 5 passed its defined milestone gate at **10.0/10.0**, above the required **9.9/10.0** cutoff, with **zero critical defects**.

## Explanation-overhead measurements
Two local measurements were recorded during verification because timing is environment-dependent even when the workload is deterministic:

- Standalone benchmark from the final verified code run: 100 samples, core mean `1.742 ms`, explanation mean `9.778 ms`, reported overhead `561.4%` of core.
- Master-exam embedded benchmark: core mean `2.376 ms`, explanation mean `3.965 ms`, reported overhead `166.89%` of core.
- A separate CI-captured standalone benchmark in the same milestone family recorded `2.506 ms` core, `4.205 ms` explanation, `167.81%` overhead.

These values are benchmark timings, not model-quality metrics. They vary with runner/environment conditions and are explicitly **not production latency claims**.

## What was implemented
- Versioned `DecisionEvidence` schema `evidence-v1`.
- Optional `DecisionResult.evidence` field preserving Phase 4 result compatibility.
- Detector-level contributions for signature, supervised, anomaly and behavioural signals.
- Feature-group attribution for supervised and anomaly models.
- Behavioural-state evidence.
- Human-readable explanations.
- Rule/model/feature/dataset/baseline/pipeline provenance.
- Numeric feature snapshots and group summaries without raw payload retention.
- Telemetry evidence under `event-v2`.
- Privacy/reproducibility/attribution/backward-compatibility tests.
- Executable master exam with a 9.9 cutoff and critical-defect rule.
- Separate explanation-overhead benchmark.
- GitHub Actions verification workflow.
- Privacy-safe Supabase migration `supabase/migrations/20260917101500_phase5_decision_evidence.sql`.
- Portable future-chat handoff archive generation.

## Privacy boundary
The Phase 5 evidence object and migration do not intentionally retain raw request payloads, raw query strings, raw headers or source IPs. The dedicated privacy tests and static migration check passed. The legacy runtime database schema may still contain payload-capable fields outside the new Phase 5 evidence table and is therefore not represented as privacy-safe Phase 5 evidence storage.

## Phase 4 regression baseline
Before Phase 5, the repository recorded **48/48 PASS** and a **10.0/10.0** Phase 4 master exam with zero critical defects. Phase 5 preserved that enforcement baseline while adding evidence after the existing policy decision.

## Historical failed verification
An earlier Phase 5 workflow execution failed two legacy config assertions because they still expected the Phase 3 default pipeline value while the Phase 5 runtime explicitly used `phase5`. The compatibility contract was restored without changing Phase 5 enforcement semantics. Subsequent real GitHub Actions execution passed the full 54-test regression and Phase 5 master exam.

## Database state
The Phase 5 Supabase migration is committed and privacy-checked but **not claimed as applied to a live Supabase database**. No live database execution result was available in the Phase 5 CI workflow. This is kept explicit so a future release can perform and record the real migration application.

## Remaining after Phase 5 milestone
- Apply the Phase 5 migration to the live connected database.
- ModSecurity/Coraza verification.
- TLS/HTTPS deployment verification.
- Optional semi-supervised evidence path.
- ML rule recommendation/approval.
- Baseline/feedback/drift/retraining.
- Production storage/auth/RBAC/data-minimization hardening.
- Full load/failure testing beyond milestone benchmarks.
- Challenge-specific scenario evidence.
- Dashboard migration.
- Five-minute demo.
- Technical docs/slides.
- Final release gate.

## Evidence boundary
The Phase 5 score is an executed repository milestone result. The ML evaluation metrics in the project remain deterministic synthetic benchmark/workload evidence and are not real-world Internet WAF accuracy claims. The explanation timing values are local benchmark measurements, not production latency guarantees.
