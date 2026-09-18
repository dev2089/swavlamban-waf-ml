# Swavlamban WAF ML

Challenge 3: ML-integrated open-source WAF.

## Current verified milestone

Phase 4 is complete on branch phase4-independent-final, based on phase3-independent-final.

Delivered milestones:
- Phase 0 freeze/baseline
- Phase 1 architecture foundation
- Phase 2 real HTTP interception and pre-forwarding enforcement
- Phase 3 bounded 40-feature http-v2 HTTP feature pipeline
- Phase 4 supervised, benign-only unsupervised anomaly and learned behavioural ML

Phase 4 uses:
- HistGradientBoostingClassifier for supervised classification
- OneClassSVM trained only on a benign baseline for anomaly detection
- LogisticRegression over bounded per-source sliding-window features for learned behavioural detection

The live EdgeWAF combines deterministic signatures with all three ML signals. Known signature hits remain hard blocks. ML inference failures are fail-closed. Model artifacts are versioned and validated against the 40-feature http-v2 contract.

## Verification

Local Phase 4 gate:
- compileall PASS
- 53/53 regression tests PASS
- supervised holdout F1 1.0, FPR 0.0, n=1500, synthetic only
- unsupervised benign FPR 0.017333, attack detection 0.888, synthetic only
- behavioural normal max 0.406313, burst final 0.980486
- artifact reproducibility PASS
- latest direct benchmark 323.94 req/s for 2000 requests
- latest E2E benchmark 161.89 req/s for 1000 requests at concurrency 50, 0 HTTP 500, 850 upstream hits
- inherited Nginx evidence: allow=200, block=403

## Reproduce

Install Phase 4 dependencies:

python -m pip install -r requirements-phase4.txt

Run:

python -m compileall -q waf tests
python -m pytest -q tests
python scripts/train_phase4_models.py --output models/phase4_models.joblib
python scripts/phase4_gate.py
python phase4_benchmark.py

## Durable project memory

The canonical future-chat context is under waf/database/.

Start with:
- waf/database/FUTURE_CHAT_START_HERE.md
- waf/database/PHASE_4_STATE.json
- waf/database/PHASE_4_INDEPENDENT_AUDIT.md
- waf/database/PROJECT_JOURNAL.jsonl
- waf/database/ALL_PHASES_TODO.md

## Overall status

Overall Challenge 3 remains IN_PROGRESS.

Still open:
- semi-supervised ML
- separate ModSecurity/Coraza verification
- full TLS termination/inspection
- richer explainability
- ML-derived rule lifecycle
- feedback/drift/retraining
- production storage/auth/RBAC/RLS/secrets
- million-request and multi-node validation
- full failure/chaos matrix
- dashboard migration
- official scenarios
- five-minute demo
- technical document/slides/reproducibility package
- final 100% release gate

Synthetic metrics are not real-world WAF accuracy claims, and local throughput is not a production capacity guarantee.
