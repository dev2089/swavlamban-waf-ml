# Phase 9 - Production Integration and Challenge Evidence

## Gate
Hard cutoff: 9.9/10.0. Any failed critical integration check fails the phase.

Phase 9 turns the Phase 8 security primitives into the production-facing FastAPI boundary and adds controlled Challenge 3 scenario/performance evidence without claiming unavailable external infrastructure.

## Implemented
- FastAPI adapter with signed bearer authentication and Phase 8 RBAC permissions.
- Operator-gated analysis, reviewer-gated rule approval, admin-gated deployment.
- Explicit CORS allowlist, security headers and request-ID handling.
- Server-only Supabase REST adapter. The service-role credential is never exposed to browser code or returned by endpoints.
- Runtime telemetry excludes raw request body, raw headers and raw query parameters and hashes source IP identifiers.
- Phase 9 Supabase migration removes permissive anonymous policies, grants browser clients read-only access through authenticated roles, keeps writes on `service_role`, scrubs old raw request fields, and adds future-write privacy checks.
- Local HTTPS termination through nginx into the live WAF edge, with TLS 1.2/1.3 and HSTS configuration.
- Deterministic Challenge 3 scenario evidence: baseline traffic, HTTPS-after-termination, known SQL/XSS, an unseen command-injection variant, and API-burst behavioural detection.
- Deterministic 500-request in-process performance measurement.

## Verification artifacts
- `phase9_master_exam_result.json`
- `phase9_scenario_evidence.json`
- `phase9_tls_evidence.json`
- `tests/test_phase9_production_api.py`
- `scripts/phase9_master_exam.py`
- `scripts/phase9_scenario_benchmark.py`
- `scripts/phase9_tls_smoke.py`
- `supabase/migrations/20260917150000_phase9_runtime_security.sql`

## Known boundary
No Supabase project credentials are present in the verification environment, so the SQL migration is a deployable path, not proof of a live database migration. nginx is locally available, but the environment has no ModSecurity/Coraza module. Performance is measured in-process and is not an Internet-scale throughput claim.
