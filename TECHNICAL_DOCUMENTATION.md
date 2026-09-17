# Swavlamban WAF ML - Technical Documentation

## 1. Problem and solution

Swavlamban WAF ML addresses Challenge 3 by combining an open-source Web Application Firewall edge with a machine-learning-assisted anomaly detection module. The implementation is organized around a real request path rather than a dashboard-only simulator.

The verified process-level path is:

```text
Client
  -> nginx + ModSecurity
  -> Swavlamban WAF gateway
  -> request normalization + HTTP feature extraction
  -> deterministic rules + ML/behaviour signals
  -> risk/decision engine
  -> ALLOW or BLOCK
  -> protected upstream when allowed
```

Decision telemetry is separated from the security decision path using a bounded asynchronous dispatcher. The operator dashboard reads controlled runtime state through the authenticated API.

## 2. ML architecture

The current detection layer uses a combination of supervised classification, unsupervised anomaly detection and stateful behavioural signals. The HTTP feature schema is versioned and contains request-structure, payload, security-semantic and behavioural features.

The decision pipeline records evidence rather than only a label. A detection can expose the model/version, feature schema, detector contributions, risk score, reason list and related managed-rule identifiers. This makes the result suitable for administrator review and later feedback.

The CI/evidence corpus is synthetic and deterministic. Metrics generated from that corpus are explicitly scoped to the test corpus and are not represented as field-traffic accuracy.

## 3. WAF and rule integration

The open-source WAF edge is represented by nginx + the ModSecurity nginx connector in the process-level evidence harness. The local TLS scenario terminates HTTPS at nginx before the request is inspected by the Swavlamban gateway.

Managed rules use the lifecycle:

```text
Detection/evidence
  -> candidate rule
  -> replay validation
  -> authenticated human approval
  -> deployment
  -> enforcement
  -> rollback
```

A candidate rule is checked against deterministic positive and negative replay cases. Deployment requires an authenticated permission. Deployment and rollback events are auditable.

## 4. Continuous learning

Learning control from the prior phases provides baseline, feedback, drift, challenger-training and explicit promotion/rollback controls. The active model is not silently replaced by an unvalidated retrained model.

Training data, model metadata, feature schema and decision evidence carry version identifiers so that an operator can determine which artifact produced a decision.

## 5. Security architecture

The runtime uses signed bearer tokens with issuer, audience, expiration, subject and role checks. Administrative actions are permission-gated, and approval identity is bound to the authenticated subject.

Persistence is server-side. Sensitive request material such as raw headers, raw query strings and raw bodies is excluded from persisted telemetry in the Phase 10 runtime path, while source identity is hashed for the stored record.

Production-sensitive configuration is supplied through environment variables. No real secret values are stored in the repository or handoff artifacts.

The live Supabase control plane is protected by RLS and the Phase 10 verification package contains the live-state audit evidence. Database migrations are deployment instructions; runtime verification remains the controlling evidence.

## 6. Performance architecture

The request-time path is intentionally lightweight. Heavy persistence, analytics and learning operations are kept out of the synchronous security decision path.

The Phase 10 load harness is configurable for duration, concurrency, request rate and payload profile and reports achieved requests/second, p50/p95/p99 latency, mean/max latency and error rate.

The measured load result is a bounded local benchmark. It is not presented as an Internet-scale capacity measurement. The architecture is designed so gateway and inference workers can be scaled horizontally, but million-request physical capacity remains a deployment-dependent projection rather than a fabricated benchmark.

## 7. Reliability

The gateway enforces body and response-size limits, rate limiting, upstream timeouts and unavailable-upstream handling. The evidence suite exercises clean startup, TLS, WAF enforcement, rule replay, bounded load, authenticated dashboard behavior and release packaging.

Failure of optional persistence must not be silently represented as a successful persistent write. Telemetry is bounded so burst conditions cannot grow an unbounded in-memory queue.

## 8. Dashboard

The authenticated operator dashboard exposes runtime security state, threat events, model/telemetry state and managed-rule lifecycle information. The process-level demo captures the dashboard at important lifecycle points and produces a deterministic 5-minute recording artifact.

The dashboard is evidence-driven. Current release documents do not use hardcoded historical threat counts or invented accuracy metrics as live runtime values.

## 9. Verification and release

The Phase 10 release workflow performs a clean checkout, installs the runtime and process-level WAF tooling, materializes the deterministic CI model artifact, compiles the implementation, performs gateway startup verification, runs the strict master exam, generates submission artifacts, builds the portable auditor handoff and records checksums.

The final evidence package includes:

- regression and compile results
- deterministic scenario evidence
- TLS evidence
- process-level ModSecurity enforcement evidence
- managed-rule replay evidence
- bounded load evidence
- dashboard/demo evidence
- requirement traceability
- production-readiness assessment
- security audit
- reliability report
- claim ledger
- negative-evidence register
- technical PDF and presentation source

## 10. Explicit evidence boundaries

The following are not claimed as verified in the free CI environment:

1. Public certificate issuance and rotation.
2. Public Internet HTTPS verification.
3. Physical Internet-scale distributed load/failure testing.
4. Field-traffic ML accuracy from the synthetic CI/evaluation corpus.

These boundaries are intentionally visible in the audit package so that design projections are not confused with measurements.
