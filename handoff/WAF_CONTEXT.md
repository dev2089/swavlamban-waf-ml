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
Phase 2 is complete on branch `phase2-final`. The live edge now inspects real HTTP requests before forwarding, blocks high-risk requests with 403, and has an automated Nginx integration gate. Phase 3 is next: production HTTP feature pipeline.

## Important honesty boundary
Nginx integration is verified. A separately installed ModSecurity/Coraza engine was not present in the terminal and is not claimed as verified. Overall Challenge 3 remains in progress.
