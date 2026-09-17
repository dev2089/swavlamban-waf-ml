# GitHub State

## Authoritative repository state
- Repository: `dev2089/swavlamban-waf-ml`
- Baseline main commit: `1cc4f91dd6828039f834ae4dc2b466191d04f229`
- Phase 1 authoritative branch: `phase1-final`
- Phase 2 authoritative branch: `phase2-final`
- Phase 3 authoritative branch: `phase3-final`
- Phase 4 authoritative branch: `phase4-final`
- Latest Phase 4 branch head before this state-record update: `ad6b0d44ac4b493cc9446e3f0c8d143695dfc8de`
- Main remains intentionally untouched by milestone work.

## Phase 4 repository state
`phase4-final` contains the Phase 4 supervised, benign-only unsupervised and learned behavioural ML implementation, live-edge integration, tests, documentation and machine-readable state. The versioned model manifest is committed; the binary model artifact is included in the portable handoff bundle and can be regenerated with `python scripts/train_phase4_models.py`.

## Verification
- Phase 4 milestone: PASS 10.0/10.0, critical defects 0.
- Full regression: 48/48 PASS.
- Latest direct ML self-test: 2,000 requests, 538.0 req/s, 1,900 allow, 100 block.
- Latest bounded E2E self-test: 1,000 requests, 449.4 req/s, 850 HTTP 200, 150 HTTP 403, 0 errors.
- Latest extended E2E: 5,000 requests, 410.4 req/s, 4,250 HTTP 200, 750 HTTP 403, 0 errors.
- Model artifact SHA-256: `bc54790f8f79daf3dc2fdc0a0a34290478e016e5d297ba8e795692435f17b571`.

## Scratch branches
Several non-authoritative scratch/verification branches were created during Git data-control experiments. Do not use them as project source of truth. Use `phase4-final` only.

## Important honesty boundary
Phase 4 milestone is complete only for its defined scope. Synthetic ML metrics/workloads are reproducibility evidence only, not real-world Internet WAF accuracy. ModSecurity/Coraza verification, TLS, optional semi-supervised work, expanded explainability, ML rule lifecycle, controlled retraining/drift, production storage/auth/RBAC, full scenario evidence, dashboard, final demo and release remain open.
