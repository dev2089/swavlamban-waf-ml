# Phase 5 Master Exam

Protocol: 100 percent gate with 9.9/10 cutoff and critical-defect failure.

Result: 10.0/10.0
Critical defects: 0

Gates:
- compileall PASS
- 63/63 full regression PASS
- 10/10 Phase 5 evidence tests PASS
- privacy/static PASS
- deterministic evidence PASS
- proxy evidence correlation PASS
- ML failure explanation PASS
- explanation benchmark PASS

Evidence:
- evidence-v1
- event-v2
- complete 40-feature http-v2 numeric snapshot
- detector contributions
- supervised/anomaly group attribution
- behavioural/signature evidence
- provenance
- deterministic explanation
- no raw payload/query/header/source-IP/host text in evidence

Latest explanation benchmark:
100 samples, core mean 3.4415 ms, evidence mean 6.0215 ms, core p50 2.5138 ms, evidence p50 4.9453 ms, core p95 7.3971 ms, evidence p95 10.6243 ms, evidence/core 174.97%.

These are local deterministic timing measurements, not production latency evidence.

Independent acceptance:
phase5-independent-final was built from phase4-independent-final. The older phase5-final line was not accepted as authoritative because it diverged and contained stale feature metadata plus a reverse-proxy buffering regression.

Overall Challenge 3 remains IN_PROGRESS.
