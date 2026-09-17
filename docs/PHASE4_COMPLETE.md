# Phase 4 Completion Record

**Project:** `swavlamban-waf-ml`
**Challenge:** Challenge 3 - ML-integrated open-source WAF
**Phase:** 4 - Supervised + unsupervised + learned behavioural ML
**Status:** PASS
**Internal phase gate:** 10.0 / 10.0

## Implemented
- Supervised `HistGradientBoostingClassifier` over `http-v2`.
- Benign-only unsupervised `OneClassSVM` with threshold learned from a benign baseline.
- Learned stateful behavioural `LogisticRegression` over per-source window features.
- Deterministic labelled training dataset and varied benign-only baseline generator.
- Versioned persisted model artifact containing all three components.
- Safe artifact validation and per-edge behavioural state isolation.
- Live EdgeWAF integration combining signature, supervised, unsupervised and behavioural signals.
- Reproducible evaluation, regression, artifact round-trip and live-edge tests.

## Final verification
- Full regression: **48/48 PASS**.
- Compileall: **PASS**.
- Master exam: **10.0/10.0**, critical defects **0**.
- Supervised: accuracy/precision/recall/F1 `1.0`; FPR `0.0`; n=`1500`; synthetic only.
- Unsupervised: FPR `0.0227`; attack detection `0.8980`; benign=`750`; attack=`3000`; synthetic only.
- Behaviour: normal max `0.406313`; burst final `0.980486`; escalated `true`.
- Extended E2E: `5000 requests, 410.4 req/s, 4250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms, 4250 upstream hits, 1000 events`.
- Artifact: `199123` bytes; SHA-256 `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.
- Secret/TODO scans: PASS.

## Failed cycles retained
- stale artifact after detector replacement → retrained and artifact-validated;
- benign false-positive regression → expanded baseline variation and retrained;
- oversized mandatory benchmark → bounded gate + standalone extended benchmark;
- learned behaviour artifact compatibility failure → artifact rebuilt and validated;
- thread-lock serialization failure → explicit pickle state excluding live windows;
- behavioural state isolation/name regression → fresh runtime state + stable `behaviour-v1`;
- master-exam import-path failure → repository-root path fix.

## Open work
ModSecurity/Coraza verification, TLS/HTTPS, optional semi-supervised path, expanded explainability, ML rule lifecycle, baseline/feedback/drift/retraining, production storage/auth/RBAC, full load/failure matrix, complete challenge scenarios, dashboard, five-minute demo, technical submission package and final release gate.

## Honesty boundary
Phase 4 is complete only for this defined milestone. Synthetic metrics must not be presented as real-world Internet WAF accuracy. Overall Challenge 3 remains IN_PROGRESS.
