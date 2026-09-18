# Phase 2 Independent Test Report

| Gate | Independent result |
|---|---|
| Python compile | PASS |
| Configuration regression | PASS |
| Feature extraction tests | PASS |
| Generic pipeline tests | PASS |
| Edge SQLi detection | PASS |
| Edge XSS detection | PASS |
| URL-decoded payload handling | PASS |
| Traversal detection | PASS |
| Command-injection detection | PASS |
| Actual upstream blocking | PASS |
| Benign upstream forwarding | PASS |
| Oversized upstream response handling | PASS |
| Nginx syntax | PASS |
| Nginx -> WAF -> upstream integration | PASS |
| Static secret scan | PASS |
| Phase-2 TODO/no-op scan | PASS |
| Deterministic demo | PASS |
| 5,000-request benchmark run 1 | PASS |
| 5,000-request benchmark run 2 | PASS |
| 5,000-request benchmark run 3 | PASS |
| 5,000-request benchmark run 4 | PASS |
| 5,000-request benchmark run 5 | PASS |

## Benchmark evidence

Five independent local runs used 5,000 requests and concurrency 100 against the Phase 2 proxy:

- Run 1: 1,916.277 req/s
- Run 2: 1,604.495 req/s
- Run 3: 1,876.252 req/s
- Run 4: 1,888.372 req/s
- Run 5: 1,907.636 req/s
- Median: 1,888.372 req/s
- Every request matched its expected allow/block result in every run.
- The in-process event history remained bounded to the latest 1,000 events.

These are local measurements, not a claim that the system has processed millions of production requests.

## Test-environment limitation

The terminal could not resolve the package index, so a fresh dependency installation inside a clean virtual environment was not possible. The terminal already contained compatible aiohttp and pytest packages, and all executable local tests were run with those installed packages. GitHub Actions independently confirmed that dependency installation, compile, the Python test suite, demo and benchmark steps succeeded on an Ubuntu runner; its Nginx integration attempt initially failed only because the runner could not write the default nginx PID path. The repository was then corrected to use a writable temporary PID path. A post-fix GitHub Actions rerun was not automatically triggered by the connector-authored commit, so the post-fix Nginx result is supported by the independent local rerun, not by a claimed CI rerun.

## Phase 2 gate

100% PASS for the defined Phase 2 acceptance criteria.

## Remaining overall Challenge 3 work

ModSecurity/Coraza engine verification, HTTPS/TLS, production feature/behaviour pipeline, ML models, explainability, ML-derived rule lifecycle, continuous learning, production auth/storage, scale testing, dashboard migration and final deliverables remain open.