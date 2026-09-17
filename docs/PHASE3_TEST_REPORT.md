# Phase 3 Test Report

| Gate | Result | Evidence |
|---|---|---|
| Compile | PASS | `python -m compileall -q waf tests` |
| Feature regression | PASS | schema, ranges, decoding, duplicates, cookies, content type, malformed input, determinism, large input, unknown method |
| Live edge regression | PASS | `tests/test_phase3_edge.py` |
| Phase 1/2 regression | PASS | existing suites run together after schema update |
| Randomized feature fuzz | PASS | 20,000 randomized HTTP-like requests, 0 exceptions |
| Feature performance | PASS | 100,000 extractions, 34,854.5 req/s |
| End-to-end WAF performance | PASS | 5,000 requests, 2,354.3 req/s |
| Secret-like scan | PASS | no secret-like values in Phase 3 code |
| TODO/no-op scan | PASS | no TODO/FIXME/pass-only stubs in Phase 3 code |

## E2E distribution

- Allowed: 4,500
- Blocked: 500
- Protected-upstream hits: 4,500
- Bounded event buffer: 1,000

## Performance note
Phase 3 deliberately does more per-request work because it extracts a richer 38-feature representation. The 2,354.3 req/s value is a local measurement, not a production capacity guarantee.

## Score

**10.0 / 10.0 for the Phase 3 acceptance criteria.**

This milestone score does not transfer to the overall challenge until later milestones are independently verified.
