# Phase 10 execution and remediation ledger

This ledger is intentionally cumulative. Earlier Phase 1-9 evidence remains authoritative for its own milestone; this phase adds the final-candidate repair history and evidence boundary.

## Phase 10 implementation cycles

1. Production runtime integration: HTTP(S) gateway, nginx + ModSecurity seam, protected-upstream enforcement, authenticated dashboard, asynchronous telemetry, rule lifecycle APIs, bounded load harness and submission-artifact tooling.
2. CI startup repair: the clean runner now materializes the deterministic Phase 4 model artifact before gateway startup so the runtime does not spend the readiness window training from scratch.
3. Gateway configuration repair: slotted-dataclass defaults were made explicit and isolated startup smoke was added.
4. TLS repair: the local self-signed certificate smoke now provisions its own log/pid paths, captures nginx config-test diagnostics, and uses curl `-k` only for the intentionally self-signed local certificate.
5. API acceptance repair: telemetry, model metadata, rule recommendation, rule validation, rule state and rollback endpoints were wired to the same runtime lifecycle; approval is bound to the authenticated subject.
6. Async persistence repair: the request path updates only a bounded local runtime view and enqueues durable telemetry; persistence is performed by the worker thread.
7. Rule replay repair: signature-dominant evidence now has an explicit evidence-feature fallback to produce an allowlisted ML-derived rule candidate rather than silently returning an empty recommendation.
8. Audit false-positive repair: the audit scanner excludes only its own source file because its detector literals are intentionally present there; all other tracked files remain scanned.
9. Gate timing repair: the master exam gives the five-minute browser demo sufficient bounded time and the GitHub job has a 20-minute hard ceiling.

## CI failure history retained

The user observed release-candidate runs of approximately 4m45s and 6m35s. Those wall-clock values include dependency installation, nginx/ModSecurity setup, Chromium installation and the broader Phase 10 gate. They are not WAF request-latency measurements. Product latency is captured by the bounded load harness and local process-level evidence.

Earlier gate failures included: fallback model startup timeout; gateway slots default conversion error; missing TLS evidence caused by a certificate-verifying preflight; incorrect aiohttp test harness; missing runtime API surfaces; empty ML-derived rule replay; five-minute dashboard demo exceeding the old 180-second subprocess ceiling; audit scanner self-match.

## Release boundary

The project does not claim public certificate issuance/rotation, Internet-scale distributed execution inside free CI, or production-field ML accuracy from synthetic evaluation. These are recorded as negative evidence rather than suppressed.
