# Phase 10 Release Candidate Failure and Repair Log

This document is cumulative and intentionally preserves defects discovered during release gating. It is not a greenwash summary.

## Why Phase 10 CI became slower
The release-candidate gate now installs nginx, ModSecurity, Chromium and FFmpeg, builds the deterministic model artifact, runs the complete regression suite, starts a real gateway, executes WAF enforcement and TLS process tests, runs a bounded network load test and records a five-minute browser demo. Its wall-clock duration is therefore not comparable to the earlier minimal test loops. WAF request latency is measured separately by `phase10_load_harness.py`.

## Failure and repair sequence
- Fallback ML training during clean CI startup consumed the readiness window. CI now materializes `models/phase4_models.joblib` first.
- Slotted `GatewayConfig` class attributes were incorrectly used as default values. Explicit numeric defaults are now used and isolated startup smoke covers the path.
- TLS smoke attempted an ordinary certificate-verifying HTTPS preflight against an intentionally self-signed local cert. The preflight was removed; curl uses `-k` only for the local test and nginx config-test diagnostics are captured.
- aiohttp integration tests passed `web.Application` directly to `AiohttpTestClient`; they now wrap it in `TestServer` first.
- Acceptance tests found missing telemetry/model/rule lifecycle API routes. The production API now exposes these surfaces and keeps approval identity tied to the authenticated subject.
- Rule replay found signature-dominant evidence could yield zero managed ML-derived recommendations. The lifecycle adds a deterministic allowlisted feature-evidence fallback, still subject to validation and human approval.
- The audit scanner found its own detector literals. It now skips only itself while scanning all other tracked repository files.
- The master exam previously limited the five-minute dashboard recording to 180 seconds. The dashboard demo check now has a 480-second bound and the GitHub job has a 20-minute maximum.

## Evidence rule
A release is not green because a generated artifact exists. The final gate must execute cleanly and all required evidence must correspond to the exact final commit.
