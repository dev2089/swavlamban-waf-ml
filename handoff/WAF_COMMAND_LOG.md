# WAF Command / Evidence Log

## Phase 1
- `python -m compileall -q waf tests` -> PASS.
- `python -m unittest discover -s tests -v` -> 13/13 PASS.
- `scripts/phase1_self_test.py` -> PASS.
- 100,000 mixed randomized HTTP-like requests -> ~23,882 decisions/sec.

## Phase 2
- Python compileall -> PASS.
- Phase 2 tests -> 6/6 PASS.
- Nginx syntax/integration -> PASS: allow=200, block=403.
- 5,000-request local E2E benchmark -> 3,477.8 req/s; 4,500 allow; 500 block; 1,000 bounded events.
- Static secret/dangerous-I/O/TODO scans -> PASS.

## Phase 3
- `python -m compileall -q waf tests` -> PASS.
- Full regression -> 34/34 PASS.
- First feature cycle -> FAIL: percent-ratio `TypeError` plus over-specific Unicode-path assertion.
- Remediation -> fixed ratio calculation and corrected semantics-preserving test.
- Header-boundary follow-up -> PASS after limiting normalized header processing to 128 headers.
- Self-test runner -> PASS.
- Randomized feature fuzz -> 20,000 requests, 0 exceptions.
- Final feature benchmark -> 100,000 extractions, 40,134.9 req/s; 38 features.
- Final E2E proxy benchmark -> 5,000 requests, 4,500 allow, 500 block, 2,599.5 req/s, 1,000 bounded events.
- Static secret-like scan -> PASS.
- New Phase 3 TODO/FIXME/pass-only scan -> PASS.

All measurements are local terminal evidence and are not production capacity guarantees.
