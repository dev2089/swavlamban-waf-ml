# Phase 4 Independent Audit

Date: 2026-09-18
Repository: swavlamban-waf-ml
Authoritative branch: phase4-independent-final
Base: phase3-independent-final

## Audit method

Earlier Phase 4 material was inspected before acceptance. Its ancestry diverged from the independently verified Phase 3 branch, and its inspected model manifest claimed 38 features while the authoritative http-v2 contract contains 40. The inspected builder tree also documented a binary artifact that was not present in that tree. The accepted implementation was therefore rebuilt from phase3-independent-final, then executed locally from a synchronized workspace.

## Phase 4 implementation

The live security seam now performs:

RequestEnvelope -> http-v2 -> deterministic signatures + supervised ML + unsupervised anomaly ML + learned behavioural ML -> deterministic edge policy.

Supervised detector: HistGradientBoostingClassifier on deterministic labelled synthetic HTTP data.

Unsupervised detector: OneClassSVM fitted only on benign baseline rows. A benign-derived threshold converts the one-class decision function to a bounded risk score.

Behavioural detector: LogisticRegression over bounded per-source sliding-window features: request count, request rate, unique-path count/ratio and inter-arrival statistics. State is process-local and capped at 128 events per source.

Model and feature contracts are versioned. The model artifact uses phase4-model-v1, model version phase4-ml-v1 and feature schema http-v2. Runtime validates the artifact and requires exactly 40 features.

Training and evaluation are outside the request-time path. No database writes occur during ML inference. Raw payloads are not put into the feature vector.

Known signature matches remain hard blocks. ML inference failures are represented as a risk=1.0 fail-closed signal so a model failure cannot silently bypass enforcement.

## Discovered defects and corrections

- Stale builder feature-count metadata: 38 -> authoritative 40.
- Divergent builder ancestry: rebuilt from phase3-independent-final.
- Unverifiable builder binary claim: switched to reproducibly generated, source-manifested artifact.
- Phase 3/4 config-test expectations: updated to pipeline_version phase4.
- Initial ML failure handling did not force block: fixed with explicit ml-runtime-failure risk=1.0 policy.
- Behavioural mutable state serialization/sharing risk: live windows are excluded from pickle state and recreated per EdgeWAF runtime.
- Potential pytest collection collision from root self-test naming: gate remains under scripts/phase4_gate.py.

## Final local evidence

- compileall: PASS
- regression: 53/53 PASS
- supervised holdout: accuracy=1.0, precision=1.0, recall=1.0, F1=1.0, FPR=0.0, n=1500
- unsupervised holdout: benign FPR=0.017333, attack detection=0.888, benign n=750, attack n=3000
- behavioural synthetic workload: normal max=0.406313, burst final=0.980486, escalated=true
- model artifact round-trip: PASS
- repeated artifact generation: same 183235-byte artifact SHA-256 8fe56e23ea19a6dc2e82563b0ae4f1d0754a3df2ee5be1e4042a51910edabf50
- latest local direct benchmark: 2000 requests, 323.94 req/s, 1900 allow, 100 block, 0 alert
- latest local E2E benchmark: 1000 requests, concurrency 50, 161.89 req/s, 850 HTTP 200, 150 HTTP 403, 0 HTTP 500, 850 upstream hits, p50 255.985 ms, p95 689.393 ms
- Nginx Phase 3 integration remained available and its configuration was not changed by Phase 4; prior verified integration was allow=200/block=403.
- static secret/stub scans: PASS
- Phase 4 gate: PASS

The final benchmark numbers are local measurements and fluctuate with runtime conditions.

## Open items

Semi-supervised ML, separate ModSecurity/Coraza verification, full TLS termination/inspection, richer explainability, ML-derived rule lifecycle, traffic feedback/drift/retraining, production storage/auth/RBAC/RLS/secrets, million-request and multi-node validation, production failure matrix, dashboard migration, official challenge scenarios, final five-minute demo, technical submission package and final project gate remain open.

## Truth boundary

Phase 4 PASS is only the defined ML milestone. Synthetic evaluation results do not establish real-world Internet WAF accuracy or production capacity. Overall Challenge 3 remains IN_PROGRESS.
