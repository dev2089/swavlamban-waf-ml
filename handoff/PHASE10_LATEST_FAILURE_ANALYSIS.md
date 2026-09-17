# Phase 10 latest failure analysis

This file records the release-candidate failure chain for transparent handoff.

## Known CI failure chain
1. Clean GitHub runners did not have the ignored model artifact, causing slow fallback training at startup. CI was changed to materialize `models/phase4_models.joblib` before startup.
2. `GatewayConfig` had a slotted-dataclass default access bug; defaults were changed to explicit literals and covered by startup regression tests.
3. Local TLS evidence originally used an ordinary certificate-verifying HTTPS probe against an intentionally self-signed certificate; the probe was removed in favor of `curl -k` and diagnostics.
4. The Phase 10 master exam initially used a 180-second timeout for the five-minute browser demo, making the release gate fail even when the demo path itself was healthy. The final gate must allow the full requested recording duration.
5. Gateway tests incorrectly wrapped an `aiohttp.web.Application` directly in `AiohttpTestClient`; the correct harness uses `TestServer` then `AiohttpTestClient(TestServer)`.
6. Release tests exposed missing production API routes for telemetry, model metadata, rule recommendation, validation, rollback and rule-state inspection. These are acceptance-test findings, not cosmetic documentation gaps.
7. Rule replay exposed that signature-dominant evidence could produce no ML-derived managed rule. The rule lifecycle therefore needs a deterministic evidence-backed fallback that remains on the allowlisted feature matcher surface and still requires human validation/approval.
8. The repository audit scanner flagged its own illustrative secret-detection strings. The scanner must exclude its own source file rather than weakening scanning for the rest of the repository.
9. The live Supabase project was migrated and then hardened. Security-advisor verification reached zero security lints after function `search_path` remediation; remaining index notices are performance INFO only.

## Important interpretation
The longer CI wall-clock times reported by the user are not WAF request latency measurements. Phase 10 installs nginx/ModSecurity/browser tooling, starts clean services, runs a bounded load test and can record a five-minute demo. Workflow duration therefore increased as the release gate became broader. Product latency is measured separately in the load-harness evidence.
