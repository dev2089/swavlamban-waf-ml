# Phase 4 Master Exam

Protocol: PHASE4 100_PERCENT GATE / critical-defect-fail
Result: 10.0/10.0
Critical defects: 0

Gates:
- compileall: PASS
- 53/53 full regression: PASS
- supervised held-out evaluation: PASS
- unsupervised held-out evaluation: PASS
- learned behavioural evaluation: PASS
- live edge benign allow: PASS
- live known-attack block: PASS
- model artifact round-trip: PASS
- artifact reproducibility: PASS
- fail-closed ML failure path: PASS
- security/stub scans: PASS

Measured evidence:
Supervised F1=1.0, FPR=0.0, n=1500 synthetic holdout.
Unsupervised FPR=0.017333, attack detection=0.888, benign n=750, attack n=3000 synthetic evaluation.
Behaviour normal max=0.406313, burst final=0.980486, escalation=true.
Latest local direct benchmark=323.94 req/s for 2000 requests.
Latest local E2E benchmark=161.89 req/s for 1000 requests at concurrency 50, 0 HTTP 500, 850 upstream hits.
Artifact=183235 bytes, SHA-256=8fe56e23ea19a6dc2e82563b0ae4f1d0754a3df2ee5be1e4042a51910edabf50.

Scope boundary:
All ML quality metrics are deterministic synthetic evidence. Throughput is a local measurement. This phase does not establish real-world WAF accuracy, TLS inspection, production-scale capacity, distributed behaviour or overall Challenge 3 completion.
