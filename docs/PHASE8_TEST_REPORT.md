# Phase 8 Test Report

## Acceptance gate
Cutoff: **9.9/10.0**. Any critical defect fails the phase regardless of arithmetic score.

## Test layers
1. Full repository regression, preserving Phase 1-7 coverage.
2. Python compile gate for `waf`, `tests`, and `scripts`.
3. Dedicated Phase 8 security tests.
4. Static security contract gate covering secret placeholders, RBAC permissions, Supabase RLS and anonymous-access revocation.

## Dedicated assertions
- Signed bearer token round-trip succeeds with valid claims.
- Signature tampering, expiry and unsupported/malformed authentication fail closed.
- Viewer cannot approve models; reviewer can approve models; admin alone manages roles/secrets/deployments.
- Production configuration rejects weak secrets, HTTP Supabase endpoints, missing service-role credentials and wildcard CORS.
- Response headers include `nosniff`, frame denial and `no-store` controls.
- Audit records contain identifiers only and exclude request bodies/headers.
- Supabase migration enables RLS and revokes anonymous access to security tables.

## Evidence boundary
No live credentials are stored in Git. The migration is committed as executable deployment SQL, but live application is intentionally not claimed until a real Supabase environment is supplied and verified.

## Carry-forward
Live API endpoint wiring, live Supabase execution, TLS/HTTPS deployment, ModSecurity/Coraza, load/failure testing and final challenge evidence remain open.
