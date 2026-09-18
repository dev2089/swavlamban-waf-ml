# WAF Handoff Context

Read waf/database/FUTURE_CHAT_START_HERE.md, WAF_PROJECT_STATE.json, waf/database/PHASE_4_STATE.json, waf/database/PHASE_4_INDEPENDENT_AUDIT.md, waf/database/PROJECT_JOURNAL.jsonl, waf/database/ALL_PHASES_TODO.md and the current requirement matrix before changing the project in a new conversation.

## Goal
Build a real demonstrable ML-augmented open-source WAF for Challenge 3 with HTTP(S) inspection, rule+ML decisions, explanations, enforcement, feedback/retraining, telemetry, dashboard and reproducible evidence.

## Constraints
- target cost: ₹0
- terminal is the local build/test lab
- GitHub is the source/control plane
- repository: dev2089/swavlamban-waf-ml
- baseline main commit: 1cc4f91dd6828039f834ae4dc2b466191d04f229

## Current milestone
Phase 4 is PASS on phase4-independent-final, based on phase3-independent-final.

## Phase 4 truth
The live path contains deterministic signatures plus supervised, benign-only unsupervised anomaly and learned behavioural detectors over the 40-feature http-v2 contract. Model inference is fail-closed, training/evaluation are outside request-time enforcement, and no database writes occur on the ML fast path.

## Critical boundary
Phase 4 metrics are deterministic synthetic/local evidence, not Internet-scale accuracy or production capacity claims. Semi-supervised ML, TLS, ModSecurity/Coraza, distributed behaviour, continuous learning, production persistence/auth, final dashboard/demo/submission and the final project gate remain open.
