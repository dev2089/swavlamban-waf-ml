# Phase 5 Test Report

| Gate | Result |
|---|---|
| Compileall | PASS |
| Full regression | 63/63 PASS |
| Phase 5 evidence coverage | 10/10 PASS |
| DecisionEvidence schema | PASS, evidence-v1 |
| Complete feature snapshot | PASS, 40 bounded numeric http-v2 features |
| Detector contributions | PASS |
| Feature-group attribution | PASS |
| Deterministic explanation | PASS |
| Privacy | PASS, no raw payload/query/header/source-IP/host in evidence |
| Telemetry | PASS, event-v2 |
| Proxy event correlation | PASS |
| ML failure explanation | PASS |
| Legacy compatibility | PASS |
| Secret/static/no-op scan | PASS |
| Explanation benchmark | PASS, 100 samples |
| Phase 5 gate | PASS, 10.0/10.0, 0 critical defects |

Latest explanation benchmark:
core mean 3.4415ms, core p50 2.5138ms, core p95 7.3971ms, evidence mean 6.0215ms, evidence p50 4.9453ms, evidence p95 10.6243ms, evidence/evidence-core overhead 174.97% of core.

Timing is environment-dependent and is not a production latency claim. No GitHub Actions PASS is claimed for the final independent branch head.
