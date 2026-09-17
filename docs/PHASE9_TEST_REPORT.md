# Phase 9 Test Report

## Security API
The dedicated suite covers missing/tampered bearer tokens, viewer/operator/reviewer/admin permission boundaries, production fail-closed configuration, response security headers, privacy-safe persistence, hashed source identity, deterministic request IDs and configuration mapping.

## Challenge scenarios
`phase9_scenario_benchmark.py` runs all scenarios through the Phase 2/3/4/5/6/7 live edge path. Evidence includes detector reasons, rule IDs, evidence schema and privacy flags.

## HTTPS integration
`phase9_tls_smoke.py` generates a temporary self-signed certificate, validates an nginx TLS configuration, starts a local upstream and WAF edge, then sends one allowed and one SQL-injection request over HTTPS. The expected edge outcomes are a successful upstream response and a WAF 403 block.

## Performance
The benchmark executes 500 in-process WAF decisions and records mean, p95, p99, min, max latency and derived requests/second. Network, TLS and upstream timing are deliberately excluded so the measurement remains attributable to the WAF decision path.

## Failure/remediation history
Phase 9 API tests initially had 3/4 failures because the auth fixture used a frozen timestamp while the live API correctly enforced expiration against current time. The fixture was corrected to issue current-time tokens. The first TLS smoke expected a 200 from `/health`, but the local test upstream did not expose that path, producing 404. The smoke was corrected to use `/`; the final TLS check produced 200 for allowed traffic and 403 for blocked SQL traffic.
