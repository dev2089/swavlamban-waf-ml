# GitHub State

## Authoritative repository state
- Repository: `dev2089/swavlamban-waf-ml`
- Baseline main commit: `1cc4f91dd6828039f834ae4dc2b466191d04f229`
- Phase 1 authoritative branch: `phase1-final`
- Phase 2 authoritative branch: `phase2-final`
- Phase 3 authoritative branch: `phase3-final`
- Phase 4 authoritative branch: `phase4-final`
- Latest Phase 4 branch head at handoff: `bb437c0b8fac3f17e946e07a874c718cad7c6a4f`
- Main remains intentionally untouched by milestone work.

## Phase 4 repository state
`phase4-final` contains the Phase 4 supervised, benign-only unsupervised and learned behavioural ML implementation, live-edge integration, tests, documentation and machine-readable state. The versioned model manifest is committed; the binary model artifact is included in the portable handoff bundle and can be regenerated with `python scripts/train_phase4_models.py`.

## Scratch branches
Several non-authoritative scratch/verification branches were created during Git data-control experiments. Do not use them as project source of truth. Use `phase4-final` only.

## Verification boundary
Phase 4 milestone is PASS 10.0/10.0 with 48/48 regression and zero critical defects. Synthetic ML metrics are reproducibility evidence only. ModSecurity/Coraza installation, TLS, optional semi-supervised work, later explainability/rule/learning/storage/dashboard/demo work and the final challenge release remain open.
