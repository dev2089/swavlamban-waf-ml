# WAF Handoff Context

Read this file, `WAF_PROJECT_STATE.json`, `handoff/WAF_BASELINE_AUDIT.md`, `handoff/WAF_MASTER_PLAN.md`, `handoff/WAF_CHANGELOG.md`, and the latest phase state/test report before changing the project in a new conversation.

## Goal
Build a real, demonstrable ML-augmented open-source WAF for Challenge 3: real HTTP(S) inspection, rule+ML decisions, explanations, actual allow/block enforcement, feedback/retraining, measurable telemetry, dashboard and reproducible evidence.

## Constraints
- target cost: ₹0;
- user primarily has phone + ChatGPT/free trials, not a personal PC workflow;
- terminal workspace is the lab/build/test environment;
- GitHub is the source/control plane;
- repository: `dev2089/swavlamban-waf-ml`;
- baseline main commit: `1cc4f91dd6828039f834ae4dc2b466191d04f229`.

## Locked challenge
Challenge 3. Do not switch unless the user explicitly reopens it.

## Quality
MASTER 9.9+ protocol applies to all deliverables. Final project requires executable evidence; critical defects fail regardless of arithmetic score.

## Current phase
Phase 4 is complete on branch `phase4-final`: supervised `HistGradientBoostingClassifier`, benign-only `OneClassSVM`, learned stateful behavioural `LogisticRegression`, live EdgeWAF ML integration and versioned artifact have passed the Phase 4 gate.

## Verified Phase 4 evidence
- full regression: 48/48 PASS;
- master exam: 10.0/10.0, critical defects 0;
- final extended local E2E: 5,000 requests, 410.4 req/s, 4,250 allow, 750 block, 0 errors;
- final artifact SHA-256: `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Important honesty boundary
Nginx/live edge enforcement is verified. A separately installed ModSecurity/Coraza engine is not claimed as verified. TLS, optional semi-supervised learning, explainability expansion, ML rule lifecycle, baseline/feedback/drift/retraining, production storage/auth, dashboard, complete challenge scenarios, final demo and final release remain open. Synthetic ML metrics are not real-world Internet WAF accuracy claims.
