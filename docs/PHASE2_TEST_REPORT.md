# Phase 2 Test Report

| Gate | Result |
|---|---|
| Python compile | PASS |
| Python Phase 2 tests | PASS: 6/6 |
| WAF config `from_env()` regression | PASS |
| SQL injection detection | PASS |
| XSS detection, including URL-decoded payload | PASS |
| Path traversal detection | PASS |
| Request-body inspection | PASS |
| Actual upstream blocking | PASS |
| Benign upstream forwarding | PASS |
| Nginx configuration syntax | PASS |
| Nginx integration | PASS |
| Secret scan | PASS |
| Dangerous-I/O scan | PASS |
| New-code TODO/no-op scan | PASS |

## End-to-end benchmark

5,000 HTTP requests were sent through the WAF reverse proxy with concurrency capped at 100:

- 4,500 allowed
- 500 blocked
- 3,477.8 requests/second measured in the terminal environment
- event buffer bounded to the latest 1,000 events

This is a reproducible local measurement for this implementation and environment, not a production capacity guarantee.

## Intermediate failures recorded

- An initial compile invocation used the wrong working directory.
- A configuration-default hazard under `dataclass(slots=True)` was caught in final review and fixed.
- An encoded XSS regression was caught and fixed by decoding the inspection target before matching.
- An initial proxy startup path returned 502 and was corrected before the final integration run.
- Final reruns passed after each remediation.

## Phase score

**10.0 / 10.0 for the Phase 2 acceptance criteria.**

This score does not transfer to the final hackathon submission. The overall Challenge 3 build remains in progress, and the external ModSecurity/Coraza engine is explicitly not claimed as verified.
