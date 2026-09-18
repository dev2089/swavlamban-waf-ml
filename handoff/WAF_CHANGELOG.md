# Change Log

## Cycle 0
- Challenge 3 locked.
- Baseline main commit: 1cc4f91dd6828039f834ae4dc2b466191d04f229.
- Prototype/production gaps recorded.

## Phase 1
- Architecture foundation and dependency-light WAF core added.
- 13/13 tests and compile/static checks passed at the Phase 1 gate.

## Phase 2
- Real HTTP reverse proxy, pre-forwarding enforcement, deterministic signatures, bounded I/O, request IDs and Nginx integration added.
- Independent regression and Nginx evidence passed.

## Phase 3
- Rebuilt on the independently verified Phase 2 branch.
- Added the authoritative 40-feature http-v2 production feature pipeline, shared decoding/normalization, bounded inspection, fuzzing and benchmarks.
- Phase 3 gate passed at 10.0/10.0.

## Phase 4
- Rebuilt independently from phase3-independent-final rather than inheriting divergent builder ancestry.
- Added supervised HistGradientBoostingClassifier, benign-only OneClassSVM, and learned per-source behavioural LogisticRegression.
- Added deterministic datasets/baselines, held-out evaluation, versioned model validation and reproducible artifact generation.
- Wired live EdgeWAF to signatures plus all three ML signals.
- ML inference failures are fail-closed; behavioural state is bounded and runtime-local.
- Corrected stale builder 38-feature metadata to authoritative 40-feature http-v2.
- Full local regression: 53/53 PASS; compileall PASS; artifact reproducibility PASS.
- Latest direct benchmark: 323.94 req/s for 2000 requests.
- Latest E2E benchmark: 161.89 req/s for 1000 requests at concurrency 50, 0 HTTP 500 and 850 protected upstream hits.
- Phase 3 Nginx evidence remains inherited because Phase 4 did not change the Nginx configuration; a later combined rerun timed out, so no fresh Phase 4-specific Nginx PASS is claimed.

## Current boundary
Authoritative branch: phase4-independent-final.
Overall Challenge 3 remains IN_PROGRESS. No later capability is to be treated as DONE without executable/reproducible evidence.

## Phase 5
- Inspected and rejected the divergent builder phase5-final as authoritative due to stale 38-feature metadata and a reverse-proxy buffering regression.
- Rebuilt Phase 5 from phase4-independent-final.
- Added evidence-v1, detector contributions, feature-group attribution, deterministic explanations, provenance, privacy-safe numeric evidence and event-v2 telemetry.
- Preserved Phase 4 enforcement semantics and fail-closed ML failures.
- Full local regression: 63/63 PASS; Phase 5 evidence tests: 10/10 PASS; compileall/privacy/static/determinism/proxy-correlation/failure-explanation all PASS.
- Latest explanation benchmark: 100 samples, core mean 8.7411ms, evidence mean 11.3495ms, evidence/core 129.84%, local timing only.
- Durable Phase 5 state, audit, manifest, journal, TODO and future-chat handoff recorded under waf/database/.

Current overall status: IN_PROGRESS. Next milestone: Phase 6 ML rule generation/validation/approval/deployment.
