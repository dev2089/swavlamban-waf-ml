# Phase 4 Completion Record

Project: swavlamban-waf-ml
Challenge: Challenge 3 - ML-integrated open-source WAF
Phase: 4 - supervised + unsupervised + learned behavioural ML
Status: PASS
Internal phase gate: 10.0 / 10.0
Branch: phase4-independent-final
Base: phase3-independent-final

## Delivered

- HistGradientBoostingClassifier supervised detector over http-v2.
- Benign-only OneClassSVM anomaly detector with learned threshold.
- Learned per-source LogisticRegression behavioural detector with bounded state.
- Deterministic held-out evaluation and explicit evidence scope.
- Versioned model artifact validation and deterministic regeneration.
- Live EdgeWAF integration of signature + all three ML detectors.
- Known-signature hard blocking preserved.
- ML inference failure is fail-closed.
- 53/53 full regression, compileall, artifact, Nginx and local performance evidence.

## Final measured evidence

Supervised: accuracy/precision/recall/F1=1.0, FPR=0.0, test n=1500, synthetic only.
Unsupervised: benign FPR=0.017333, attack detection=0.888, benign n=750, attack n=3000, synthetic only.
Behaviour: normal max=0.406313, burst final=0.980486, escalated=true.
Latest local direct benchmark: 2000 requests at 323.94 req/s, 1900 allow, 100 block, 0 alert.
Latest local E2E benchmark: 1000 requests at concurrency 50, 161.89 req/s, 850 HTTP 200, 150 HTTP 403, 0 HTTP 500, 850 upstream hits, p50 255.985 ms, p95 689.393 ms.
Artifact regeneration matched exactly at 183235 bytes, SHA-256 8fe56e23ea19a6dc2e82563b0ae4f1d0754a3df2ee5be1e4042a51910edabf50.

## Corrections retained in project history

Earlier Phase 4 material had divergent ancestry, stale 38-feature metadata and an unverified binary claim. The independent implementation corrected those. A first ML failure-policy implementation did not force block and was corrected. Behavioural state serialization/isolation and pytest collection hazards were also corrected.

## Remaining work

Semi-supervised ML; ModSecurity/Coraza verification; TLS; richer explainability; ML rule recommendation/approval/deployment; continuous feedback/drift/retraining; production storage/auth/RBAC/RLS/secrets; million-request/multi-node testing; full failure/chaos matrix; dashboard; official challenge scenarios; five-minute demo; technical document and slides; final release gate.

## Honesty boundary

All Phase 4 quality and throughput numbers above are deterministic local evidence. They are not Internet-scale accuracy or production-capacity claims. Overall Challenge 3 remains IN_PROGRESS.
