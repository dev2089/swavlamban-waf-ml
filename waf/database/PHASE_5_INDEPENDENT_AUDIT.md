# Phase 5 Independent Audit

Date: 2026-09-18
Authoritative branch: phase5-independent-final
Base: phase4-independent-final

## Why an independent branch

The existing phase5-final material was inspected before acceptance. It was not promoted because it diverged from the independently verified Phase 4 lineage, documented a stale 38-feature model manifest against the authoritative 40-feature http-v2 contract, and its reverse proxy regressed from the bounded response reader. Phase 5 was therefore rebuilt from phase4-independent-final.

## Delivered

DecisionEvidence evidence-v1 is generated after the Phase 4 policy decision. It records detector-level contributions, feature-group summaries, supervised/anomaly group perturbation attribution, behavioural evidence, provenance, a deterministic human explanation and a numeric snapshot of all 40 http-v2 features.

Privacy contract: raw payload, raw query, raw headers, source IP and host text are not retained in the evidence object. The snapshot contains only bounded numeric feature values.

Telemetry is event-v2 and embeds evidence when present. Existing enforcement thresholds and known-signature hard blocking are unchanged.

ML inference failure remains fail-closed and is also explainable without requiring trained model objects.

## Failures and fixes

1. Deliberate BrokenML test initially made evidence construction fail because model components were absent. Added safe fallback attribution and a fail-closed explanation.
2. Telemetry test initially found event-v1. Advanced the event schema to event-v2 and included evidence.
3. Phase 4 legacy config assertions expected the old default. Runtime and current tests were advanced to phase5 while preserving http-v2.
4. Evidence is now required to contain exactly 40 features, all bounded to [0,1].
5. Temporary smoke-file creation was removed before acceptance.

## Final local verification

- compileall: PASS
- full regression: 63/63 PASS
- Phase 5 evidence coverage: 10/10 PASS
- privacy/static scan: PASS
- deterministic evidence comparison: PASS
- proxy telemetry correlation: PASS
- ML failure explanation: PASS
- explanation benchmark: 100 samples, core mean 8.7411 ms, evidence mean 11.3495 ms, evidence/core 129.84%, local timing only
- Phase 5 gate: PASS, 10.0/10.0 equivalent, zero critical defects

## Boundary

Attribution is deterministic group-level perturbation, not a causal feature-importance claim. Timing is environment dependent. Production persistence/auth/RBAC, large-scale failure testing, TLS, semi-supervised ML, ModSecurity/Coraza and final submission work remain open. Overall Challenge 3 is IN_PROGRESS.
