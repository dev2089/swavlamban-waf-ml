# Swavlamban WAF ML - START HERE

## Current control point

The authoritative development branch is `phase10-final`.

The latest verified GitHub Actions release candidate executed the strict Phase 10 gate successfully. The exact commit SHA and workflow run are recorded in `artifacts/final_commit_sha.txt` and the workflow metadata. The repository gate reports 95/95 regression tests passing and a 100% result for the defined release checks.

## Future ChatGPT continuation

Read these in order:

1. `WAF_PROJECT_STATE.json`
2. `handoff/START_HERE.md`
3. `handoff/PHASE10_FINAL_STATUS.md`
4. `handoff/WAF_CONTEXT.md`
5. `handoff/PHASE10_EXECUTION_LEDGER.md`
6. `phase10_master_exam_result.json`
7. `phase10_demo_evidence.json`
8. `phase10_tls_evidence.json`
9. `phase10_waf_enforcement_evidence.json`
10. `phase10_rule_replay_evidence.json`
11. `phase10_load_evidence.json`
12. `handoff/PHASE10_REQUIREMENT_TRACEABILITY.json`
13. `handoff/PHASE10_NEGATIVE_EVIDENCE.md`
14. `handoff/PHASE10_CLAIM_LEDGER.md`

Treat all PASS labels as evidence to be independently reproducible, not as authority by themselves.

## Runtime architecture

The current active path is the `waf/` package. The open-source WAF edge is represented by nginx + ModSecurity in the process-level test harness. The Swavlamban gateway performs request analysis, enforcement and controlled forwarding to an upstream. Telemetry is bounded/asynchronous and the operator dashboard is served from the production API.

## Evidence boundaries

Measured local evidence is limited to the environment and scenarios actually executed. The project does not claim Internet-scale distributed load, public certificate issuance/rotation, or field-traffic ML accuracy from the synthetic evaluation corpus.

## Repository hygiene

Historical phase documents are retained for traceability. Runtime code and current user-facing documentation must describe only the verified architecture and measured evidence. Legacy prototype entry points must delegate to the canonical runtime rather than implement a second demo application.
