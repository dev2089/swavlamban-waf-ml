# Phase 4 Test Report

| Gate | Result | Evidence |
|---|---|---|
| compileall | PASS | python -m compileall -q waf tests |
| full regression | PASS | 53/53 tests |
| supervised holdout | PASS | F1 1.0, FPR 0.0, n 1500, synthetic |
| unsupervised holdout | PASS | benign FPR 0.017333, attack detection 0.888, synthetic |
| behavioural workload | PASS | normal max 0.406313, burst final 0.980486 |
| artifact round-trip | PASS | schema/version/40-feature validation |
| artifact reproducibility | PASS | two generations matched 183235-byte SHA-256 |
| ML failure path | PASS | inference failure maps to risk 1.0 and BLOCK |
| live known attack | PASS | signature layer blocks before upstream |
| Nginx | PASS/inherited | Phase 3 verified allow=200, block=403; Phase 4 did not modify Nginx config |
| direct benchmark | MEASURED | 2000 requests, latest 323.94 req/s |
| E2E benchmark | MEASURED | 1000 requests, latest 161.89 req/s, 0 HTTP 500 |
| secret scan | PASS | no scoped secret signature |
| stub/no-op scan | PASS | no scoped TODO/FIXME/pass stub |

All ML metrics use deterministic synthetic data. Throughput is a local runtime measurement.
