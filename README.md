# Swavlamban WAF ML

Challenge 3: ML-integrated open-source WAF.

## Verified milestones

- Phase 0: baseline/freeze
- Phase 1: architecture foundation
- Phase 2: real HTTP interception, pre-forwarding enforcement and Nginx integration
- Phase 3: bounded 40-feature http-v2 feature pipeline
- Phase 4: supervised, benign-only unsupervised and learned behavioural ML
- Phase 5: structured explainability and privacy-safe decision evidence

## Phase 5

DecisionEvidence schema: evidence-v1.
Telemetry schema: event-v2.

Every live decision can carry:
- detector-level contribution records
- feature-group summaries over all 40 http-v2 features
- deterministic supervised/anomaly group attribution
- behavioural/signature evidence
- human-readable explanation
- pipeline/model/schema/dataset/baseline/ruleset provenance
- bounded numeric feature snapshot

Raw payload, query, headers, source IP and host text are not retained in the evidence object.

Known attack signatures remain hard blocks. ML failures remain fail-closed. Evidence is generated after policy evaluation.

## Reproduce

python -m pip install -r requirements-phase4.txt
python -m compileall -q waf tests
python -m pytest -q tests
python phase5_explainability_benchmark.py
python scripts/phase5_gate.py

Final local gate:
63/63 tests PASS, 10/10 Phase 5 evidence tests PASS, compileall PASS, privacy/static PASS, deterministic evidence PASS, proxy correlation PASS. Latest benchmark sample: core mean 8.7411ms, evidence mean 11.3495ms. Timing is local only.

## Future-chat memory

Read waf/database/FUTURE_CHAT_START_HERE.md first, then WAF_PROJECT_STATE.json, the Phase 5 state/audit/evidence manifest, PROJECT_JOURNAL.jsonl, ALL_PHASES_TODO.md and the next-phase handoff.

## Overall status

Overall Challenge 3 remains IN_PROGRESS.

Open: Phase 6-14, semi-supervised ML, TLS/HTTPS deployment verification and separate ModSecurity/Coraza verification. Production storage/auth/RBAC, distributed scale/failure testing, dashboard, final demo and submission package are not yet complete.

Synthetic ML metrics are benchmark evidence only. Local timing is not a production capacity claim.
