# Challenge Requirement Mapping

| Need | Status after Phase 3 | Evidence / next work |
|---|---|---|
| ML integrated with open-source WAF | PARTIAL | Live WAF + ML-compatible contracts; production ML pending |
| HTTP(S) analysis | PARTIAL | Live HTTP interception + http-v2; HTTPS/TLS termination remains |
| traffic baselining | NOT DONE | Learning/baseline phase |
| behavioural analysis | NOT DONE | Behavioural detector phase |
| anomaly detection | PARTIAL | Signature foundation only; ML anomaly detector pending |
| dashboard | NOT DONE | Legacy dashboard exists; final migration pending |
| supervised/unsupervised/semi-supervised ML | NOT DONE | Production ML phase |
| explainability | FOUNDATION | Reasons/rule IDs; richer model explanations pending |
| rule recommendation | FOUNDATION | Rule lifecycle seam; ML-derived recommendation pending |
| low latency/high throughput | MEASURED LOCALLY | Feature and E2E benchmarks; production-scale proof pending |
| continuous learning/retraining | NOT DONE | Feedback/drift/retraining phase |
| logs/metrics/reports | PARTIAL | Event schema + bounded edge history; durable telemetry pending |
| demo | NOT DONE | Final scenario/demo package pending |

## Phase 3 completed

- Versioned http-v2 feature contract.
- Exactly 40 normalized numeric features.
- Safe path/query URL decoding with up to 3 passes.
- Unicode NFKC normalization.
- Bounded query parsing, headers and body inspection.
- Encoding anomaly, entropy, shape and security-indicator features.
- Edge rules share feature-pipeline normalization.
- Feature, fuzz, live-edge, regression and performance evidence.
- Durable project state, feature manifest, independent audit and journal.

## Critical honesty boundary

A requirement becomes DONE only when the intended final capability has executable or reproducible evidence. Contracts, placeholders and UI alone do not count. Overall Challenge 3 remains IN_PROGRESS.