# Challenge 3 Requirement Mapping

| Requirement | Current candidate status | Executable evidence / limitation |
|---|---|---|
| ML integrated with an open-source WAF | **PASS** | Real request path through nginx + ModSecurity + Swavlamban gateway; process-level enforcement evidence. |
| Inbound HTTP inspection | **PASS** | `waf/gateway/proxy.py`, `waf/edge/pipeline.py`; gateway/WAF E2E tests. |
| Outbound HTTP(S) content inspection | **PASS** | `waf/ml/outbound.py`, gateway response inspection, `scripts/phase10_outbound_e2e.py`; raw response is not persisted. |
| Traffic baselining | **PASS, bounded scope** | Benign-only one-class baseline plus Phase 7 baseline/feedback/drift controls; deterministic local evidence. |
| Behavioral analysis | **PASS, deterministic scenario scope** | Stateful per-source behavioural detector with API/burst scenario evidence. |
| Supervised ML | **PASS** | `HistGradientBoostingClassifier`; executable train/test metrics. |
| Unsupervised ML | **PASS** | Benign-only `OneClassSVM`; executable FPR/detection evaluation. |
| Semi-supervised ML | **PASS** | `SelfTrainingClassifier` over partially labeled data; 30% labeled training contract and executable evaluation. |
| Explainability | **PASS** | Structured evidence includes detector contributions, feature groups/attribution, reasons, rule IDs and versions. |
| Real-time anomaly detection | **PASS** | Live EdgeWAF inference before upstream forwarding; anomaly-style scenario plus process-level block tests. |
| ML output integrated into rules | **PASS** | Evidence-driven rule generation, replay validation, approval and deployment lifecycle. |
| Human approval for rule deployment | **PASS** | Explicit reviewer/approver identity and lifecycle tests. |
| TLS/encrypted traffic | **PASS, local termination scope** | nginx TLS termination and blocked/allowed HTTPS evidence. Public certificate issuance/rotation is external. |
| API abuse / bot behavior | **PASS, deterministic workload scope** | Learned behavioural detector and repeat/burst scenarios. |
| Zero-day resilience approach | **PASS as resilience mechanism, not magic detection** | Unknown/irregular behavior is surfaced through anomaly/baseline signals; synthetic proof and explicit non-claim. |
| Continuous learning | **PASS** | Phase 7 baseline, feedback, drift, challenger retraining, promotion and rollback controls. |
| Feedback loop | **PASS** | Human-reviewed feedback schema and model lifecycle tests. |
| Safe retraining/promotion | **PASS** | Challenger validation is required; active model is not silently replaced. |
| Asynchronous / non-critical telemetry path | **PASS** | Bounded telemetry path with server-side storage adapter; request decision does not require synchronous Supabase success. |
| Production storage | **PASS, live Supabase scope** | Server-only REST adapter, migrations, RLS/privilege verification and security advisor results. |
| Authentication / RBAC | **PASS** | Signed bearer token verification, role map, authenticated administrative actions. |
| Privacy / secret handling | **PASS** | Raw payload/query/header exclusion from persisted runtime evidence; secret scan; production config checks. |
| Dashboard | **PASS** | Authenticated dynamic dashboard backed by runtime API, rule/model/telemetry views. |
| Performance / low latency | **PASS, bounded local scope** | In-process and network load harness with achieved rate and p50/p95/p99; no Internet-scale claim. |
| Reliability / failure behavior | **PASS, tested scenarios** | Startup readiness, limits, upstream failure handling, queue/storage behavior, auth and reconnect tests. |
| 5-minute demo | **PASS** | Generated browser video with metadata and six dashboard screenshots. |
| Technical documentation | **PASS** | Phase 10 technical report source + generated PDF in release artifact. |
| 8-10 slide presentation | **PASS** | Presentation source and binary PPTX artifact. |
| Build/reproducibility package | **PASS** | Clean-checkout CI gate, dependency inventory, exact commit capture, source archive and manifests. |
| Internet-scale millions of requests | **NOT CLAIMED** | Architecture is scale-out friendly, but free/local CI does not physically prove Internet-scale capacity. |
| Public HTTPS certificate lifecycle | **NOT VERIFIED** | Local TLS is verified; public issuance/rotation requires external infrastructure. |
| Venue-specific deployment/submission portal operations | **EXTERNAL / OPEN** | Not part of repository correctness proof. |

## Status discipline
A requirement is **PASS** only where the repository and executable evidence support it. Design-only projections and externally constrained operations remain explicitly separated from measured facts. Historical Phase 1-9 documents retain their original milestone status and are not retroactively rewritten.
