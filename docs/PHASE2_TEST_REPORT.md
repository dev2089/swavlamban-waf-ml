# Phase 2 Test Report

| Gate | Result |
|---|---|
| Python compile | PASS |
| Python Phase 2 tests | PASS: 5/5 |
| SQL injection detection | PASS |
| XSS detection, including URL-decoded payload | PASS |
| Path traversal detection | PASS |
| Request-body inspection | PASS |
| Actual upstream blocking | PASS |
| Benign upstream forwarding | PASS |
| Nginx integration | PASS |
| Secret scan | PASS |
| Dangerous-I/O scan | PASS |
| New-code TODO/no-op scan | PASS |

## End-to-end benchmark

5,000 HTTP requests were sent through the WAF reverse proxy with concurrency capped at 100:

- 4,500 allowed
- 500 blocked
- 3,096.7 requests/second measured in the terminal environment
- event buffer bounded to the latest 1,000 events

This is a reproducible local measurement for this implementation and environment, not a production capacity guarantee.

## Phase score

**10.0 / 10.0 for the Phase 2 acceptance criteria.**

The score does not transfer to the final hackathon submission. Later phases still have explicit acceptance gates.
