# Challenge Requirement Mapping

| Need | Status after Phase 5 | Evidence / next work |
|---|---|---|
| ML integrated with open-source WAF | PARTIAL | Live WAF has signatures + supervised/unsupervised/behavioural ML; separate ModSecurity/Coraza verification remains |
| HTTP(S) analysis | PARTIAL | HTTP inspection verified; full TLS termination/inspection remains |
| traffic baselining | PARTIAL | benign training baseline exists; production traffic baseline/feedback remains |
| behavioural analysis | DONE | learned bounded per-source detector |
| anomaly detection | DONE | benign-only OneClassSVM with held-out evaluation |
| dashboard | NOT DONE | final migration remains |
| supervised/unsupervised/semi-supervised ML | PARTIAL | supervised + unsupervised implemented; semi-supervised remains |
| explainability | DONE | evidence-v1 with detector contributions, group attribution and provenance |
| rule recommendation | FOUNDATION | evidence exists; ML-derived rule lifecycle is Phase 6 |
| low latency/high throughput | MEASURED LOCALLY | local benchmarks only |
| continuous learning/retraining | NOT DONE | Phase 7 |
| logs/metrics/reports | PARTIAL | event-v2 evidence integration; durable production telemetry remains |
| demo | NOT DONE | final scenario/demo remains |

Phase 5 delivered evidence-v1, complete 40-feature numeric snapshot, detector contributions, supervised/anomaly group attribution, behaviour/signature evidence, human explanation, provenance, privacy contract and event-v2 telemetry.

Phase 5 completion does not imply overall Challenge 3 completion or production-scale/model-accuracy guarantees.