# Change Log

## Cycle 0
- Challenge 3 locked after the documented comparison.
- Baseline main commit recorded: `1cc4f91dd6828039f834ae4dc2b466191d04f229`.
- Prototype/production gap and major ML, WAF, security and documentation mismatches recorded.

## Phase 1
- Added dependency-light `waf/` security core and tests.
- Added architecture, state and handoff documentation.
- Added portable SQLite project ledger.
- Local validation: compileall PASS, 13/13 tests PASS, static scans PASS, mixed fuzz benchmark PASS.
- Phase 1 gate: 10.0/10.0.

## Phase 2
- Added live HTTP enforcement path and Nginx integration.
- Added deterministic URL-decoded SQLi, XSS, traversal and command-injection signatures.
- Added real pre-forwarding block enforcement with HTTP 403.
- Added bounded request/response sizes, timeout handling, request IDs and WAF decision headers.
- Phase 2 gate: 10.0/10.0 for its acceptance criteria.

## Phase 3
- Added production `http-v2` HTTP feature pipeline with 38 bounded normalized features.
- Added safe path/query-specific URL decoding, NFKC normalization, query limits, header normalization and body scan limits.
- Switched edge WAF and compatibility extractor to v2 and updated regression tests.
- Added comprehensive feature, fuzz, benchmark and live-edge evidence.
- Initial cycle failed on a list/ratio TypeError and an over-specific Unicode-path expectation; both were diagnosed and fixed.
- Follow-up header-boundary test exposed unbounded header iteration; normalized header processing was capped at 128 and retested.
- Final Phase 3 evidence: full regression 34/34 PASS; feature fuzz 20,000 requests/0 exceptions; latest feature benchmark 100,000 extractions at 33,527.0 req/s; latest E2E 5,000 requests at 2,764.7 req/s with 4,500 allow, 500 block and 1,000 bounded events.
- Phase 3 gate: 10.0/10.0 for its acceptance criteria.

## Important boundary
- Main branch remains untouched by milestone work.
- ModSecurity/Coraza installation, TLS, production ML, behavioural learning, production storage/auth, dashboard, complete scenario evidence, demo and final release remain open.
- Overall Challenge 3 remains in progress.
