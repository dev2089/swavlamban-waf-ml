# Phase 4 Test Report

## Final gate
**PASS: 10.0 / 10.0** against cutoff 9.9, with zero critical defects.

| Gate | Result | Evidence |
|---|---|---|
| Compile | PASS | `python -m compileall -q waf tests` |
| Full regression | PASS | `48/48` tests |
| Supervised ML | PASS | held-out deterministic synthetic HTTP benchmark |
| Unsupervised ML | PASS | benign-only baseline + attack challenge set |
| Learned behaviour ML | PASS | normal-vs-burst behavioural workload |
| Live edge integration | PASS | benign allow + known attack block + unseen anomaly alert |
| Artifact round-trip | PASS | save/load and detector presence/type assertions |
| Feature safety | PASS | 20,000-request fuzz retained from Phase 3 |
| Performance | PASS | direct, bounded E2E and extended E2E benchmarks |
| Secret/TODO scans | PASS | no secret-like values or no-op stubs in Phase 4 code |
| Handoff reproducibility | PASS | self-contained state, SQLite ledger and model artifact |

## ML evidence

### Supervised
Accuracy `1.0`, precision `1.0`, recall `1.0`, F1 `1.0`, FPR `0.0`, test samples `1500`.
Scope: deterministic synthetic HTTP benchmark only.

### Unsupervised
FPR `0.0227`, attack detection rate `0.8980`, benign evaluation `750`, attack evaluation `3000`.
Scope: deterministic synthetic HTTP benchmark only.

### Learned behaviour
Normal max score `0.406313`, burst final score `0.980486`, escalation `true`, normal requests `10`, burst requests `40`.
Scope: deterministic synthetic behavioural workload.

## Runtime evidence
- Direct ML final self-test rerun: `2000 requests, 529.6 req/s, 1900 allow, 100 block, 0 alert`.
- Mandatory bounded E2E final self-test rerun: `1000 requests, 400.2 req/s, 850 HTTP 200, 150 HTTP 403, 0 errors, p50 64.240 ms, p95 81.686 ms`.
- Retained extended E2E evidence: `5000 requests, 410.4 req/s, 4250 HTTP 200, 750 HTTP 403, 0 errors, p50 123.614 ms, p95 160.959 ms, 4250 upstream hits, 1000 events`.

## Final rerun
After all Phase 4 documentation, state and handoff synchronization, the current workspace was rerun end-to-end. `48/48` regression tests passed, compileall passed, the master exam passed at 10.0/10.0 with zero critical defects, and `phase4_self_test.py` passed. The latest rerun figures above are the current terminal evidence.

## Model artifact
- Version: `phase4-model-v1` / model `phase4-ml-v1`.
- Feature schema: `http-v2`.
- Feature count: `38`.
- Size: `199123` bytes.
- SHA-256: `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Honesty boundary
This report certifies only the Phase 4 milestone. Synthetic metrics are reproducibility evidence, not Internet-scale or real-world WAF accuracy claims. ModSecurity/Coraza, TLS, optional semi-supervised learning, explainability expansion, rule recommendation/approval, controlled retraining/drift, production storage/authentication, full challenge scenarios, dashboard migration, final demo and final submission remain later work.
