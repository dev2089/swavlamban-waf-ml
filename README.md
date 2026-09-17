# Swavlamban WAF ML

ML-integrated open-source WAF for **Swavlamban 2025 Hackathon Challenge 3**.

## Current candidate
Branch: `phase10-final`.

The latest clean-checkout GitHub Actions release run executed the full Phase 10 gate successfully, including regression tests, compilation, WAF tooling, gateway startup, TLS evidence, open-source WAF enforcement, rule replay, bounded load testing, dashboard/demo evidence, audit manifest generation, binary artifact generation, and portable handoff packaging.

The exact run and commit must be treated as the source of truth. The master exam currently records **95/95 regression tests passing** and a **100% Phase 10 release-gate result** for its defined checks.

## Architecture

```text
Client
  -> nginx + ModSecurity
  -> Swavlamban WAF gateway
  -> feature extraction + rules + ML + behavioural analysis
  -> decision
  -> protected upstream when allowed

Decision telemetry
  -> bounded asynchronous dispatcher
  -> server-side persistence / metrics
  -> authenticated operator dashboard

Reviewed feedback
  -> baseline/drift/challenger evaluation
  -> explicitly promoted model/rule versions
```

The request path is designed so the critical security decision does not require a synchronous database write.

## Read-first continuation

1. `WAF_PROJECT_STATE.json`
2. `handoff/START_HERE.md`
3. `handoff/PHASE10_FINAL_STATUS.md`
4. `handoff/PHASE10_EXECUTION_LEDGER.md`
5. `phase10_master_exam_result.json`
6. `phase10_demo_evidence.json`
7. `phase10_tls_evidence.json`
8. `phase10_waf_enforcement_evidence.json`
9. `phase10_rule_replay_evidence.json`
10. `phase10_load_evidence.json`

## Reproduce

```bash
python -m pytest -q
python -m compileall -q waf tests scripts
python scripts/phase10_demo.py
python scripts/phase10_rule_replay.py
python scripts/phase10_tls_e2e.py
python scripts/phase10_master_exam.py
```

For process-level open-source WAF enforcement, use the documented nginx/ModSecurity test harness in `scripts/phase10_waf_enforcement_e2e.py`.

## Evidence boundaries

The repository deliberately does **not** claim:

- public certificate issuance or rotation
- public Internet HTTPS verification
- physical Internet-scale distributed capacity
- field accuracy from the synthetic evaluation corpus

The bounded local load evidence reports measured throughput/latency only for the tested environment. Million-request capacity is an architecture/scaling target, not a fabricated benchmark result.

The final handoff package preserves measured results, simulated data boundaries, design projections, external constraints, claim ledger and negative-evidence register separately.
