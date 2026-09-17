# Change Log

## Cycle 0
- Challenge 3 locked after the documented comparison.
- Baseline main commit recorded: `1cc4f91dd6828039f834ae4dc2b466191d04f229`.
- Prototype/production gap and major ML, WAF, security and documentation mismatches recorded.

## Phase 1
- Added dependency-light `waf/` security core and tests.
- Added architecture, state and handoff documentation.
- Added portable SQLite project ledger in the handoff bundle.
- Local validation: compileall PASS, 13/13 tests PASS, static scans PASS, mixed fuzz benchmark PASS.
- Phase 1 gate: 10.0/10.0.
- Legacy runtime intentionally remained beside the new core pending migration.

## Phase 2
- Added `waf/edge/` live HTTP enforcement path.
- Added deterministic URL-decoded SQLi, XSS, path traversal and command-injection WAF signatures.
- Added real pre-forwarding block enforcement with HTTP 403.
- Added bounded request/response sizes, timeout handling, request IDs and WAF decision headers.
- Added Nginx integration configuration and automated Nginx gate.
- Fixed initial config default, URL-decoding and response-header defects during testing.
- Local validation: 5/5 Phase 2 tests PASS; Nginx integration PASS; compileall PASS; security/static scans PASS.
- End-to-end local benchmark: 5,000 requests, 3,096.7 req/s, 4,500 allow, 500 block.
- Phase 2 gate: 10.0/10.0 for the defined Phase 2 acceptance criteria.

## Important boundary
- Main branch has not been modified by this work.
- A separately installed ModSecurity/Coraza engine was not present in the terminal and is not claimed as verified.
- Overall Challenge 3 remains in progress.
