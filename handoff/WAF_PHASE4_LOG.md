# WAF Phase 4 Log

Project: swavlamban-waf-ml
Branch: phase4-independent-final
Base: phase3-independent-final
Status: PASS
Gate: 100%
Score: 10.0/10.0

Phase 4 introduced the first real ML inference layer on the canonical WAF security seam.

Completed:
- supervised HistGradientBoostingClassifier;
- benign-only OneClassSVM anomaly detector;
- learned per-source LogisticRegression behavioural detector;
- deterministic training data and benign baseline generators;
- held-out evaluation and scope labels;
- model artifact version/schema validation;
- live EdgeWAF integration;
- fail-closed inference error handling;
- bounded behavioural state and runtime isolation;
- regression, artifact reproducibility, performance and inherited Nginx evidence.

Corrections:
- builder 38-feature metadata corrected to authoritative 40-feature http-v2;
- divergent builder ancestry replaced with phase3-independent-final base;
- unverified builder binary claim replaced by reproducible generated artifact policy;
- ML failure path corrected from non-blocking weighted risk to risk=1.0 fail-closed;
- Phase 3 config expectations updated for Phase 4;
- pytest naming/collection hazard avoided.

Final evidence:
53/53 tests, compileall PASS.
Supervised: F1 1.0, FPR 0.0, n 1500, synthetic only.
Unsupervised: FPR 0.017333, detection 0.888, benign 750, attack 3000, synthetic only.
Behaviour: normal max 0.406313, burst final 0.980486, escalated true.
Artifact: 183235 bytes, SHA-256 8fe56e23ea19a6dc2e82563b0ae4f1d0754a3df2ee5be1e4042a51910edabf50.
Latest local direct benchmark: 323.94 req/s for 2000 requests.
Latest local E2E benchmark: 161.89 req/s for 1000 requests at concurrency 50, 0 HTTP 500, 850 upstream hits.
Phase 4 gate: PASS.

Inherited Nginx evidence: Phase 3 previously verified allow=200/block=403. No Phase 4 Nginx configuration change. A later combined rerun timed out during that script, so no fresh Phase 4-specific Nginx pass is claimed.

Carry-forward:
semi-supervised ML, TLS, separate ModSecurity/Coraza verification, richer explainability, rule lifecycle, production feedback/drift/retraining, durable storage/auth/RBAC, scale/failure matrix, dashboard, official scenarios, demo, submission package and final gate.

Overall project remains IN_PROGRESS.
