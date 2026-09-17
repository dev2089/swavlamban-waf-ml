# START HERE - Swavlamban WAF

Read these files before making any future change:

1. `handoff/WAF_CONTEXT.md`
2. `WAF_PROJECT_STATE.json`
3. `handoff/WAF_MASTER_PLAN.md`
4. `handoff/WAF_CHANGELOG.md`
5. `handoff/WAF_STATE_PHASE2.json`
6. `handoff/WAF_PHASE2_LOG.md`
7. `docs/PHASE2_COMPLETE.md`
8. `docs/PHASE2_TEST_REPORT.md`
9. `docs/ARCHITECTURE.md`
10. `handoff/WAF_REQUIREMENTS_MATRIX.md`

## Current truth
Phase 2 is PASS on branch `phase2-final`.

The verified Phase 2 edge is a real HTTP reverse proxy that inspects requests before forwarding, blocks high-risk traffic with 403, and can sit behind Nginx. The phase test includes direct enforcement and Nginx integration.

## Next phase
Phase 3: production HTTP feature pipeline.

## Rules for future ChatGPT conversations
- Do not trust old README marketing claims over executable evidence.
- Keep Challenge 3 locked unless the user explicitly reopens it.
- Use terminal as the lab and GitHub as the source/control plane.
- Apply the MASTER 9.9+ protocol to every phase and every artifact.
- Critical defects fail a phase regardless of the arithmetic score.
- Do not claim ModSecurity/Coraza, TLS, production ML, or later features until they have explicit evidence.
