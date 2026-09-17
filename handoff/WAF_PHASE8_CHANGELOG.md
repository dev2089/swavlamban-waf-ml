# Phase 8 Changelog

- Added `waf/security/auth.py` (`auth-v1`).
- Added `waf/security/rbac.py` with explicit control-plane permissions.
- Added `waf/security/secrets.py` for environment-only secret handling and redaction.
- Added `waf/storage/persistent.py` (`storage-v1`) for durable structured evidence, privacy rejection and retention.
- Updated `waf/core/config.py` with environment, storage, auth and retention controls.
- Updated `waf/storage/__init__.py` exports.
- Removed the legacy hard-coded Flask secret in `app.py`.
- Replaced `.env.example` with non-secret configuration plus secret-injection comments.
- Added `supabase/migrations/20260917130000_phase8_security_hardening.sql`.
- Added `tests/test_phase8_security_hardening.py`.
- Added `scripts/phase8_master_exam.py` and `phase8_master_exam_result.json`.
- Added Phase 8 ledger recorder and handoff builder.
- Updated project state, master plan, next-phase pointer and future-chat handoff documentation.

## Verification
- Full regression: 71/71 PASS.
- Phase 8 focused: 7/7 PASS.
- Phase 7 focused: 5/5 PASS.
- Compile gate: PASS.
- Master exam: 10.0/10.0, cutoff 9.9, zero critical defects.

## Remaining
Live Supabase application/migration, hosted identity integration, ModSecurity/Coraza, TLS, production load/failure testing, challenge-specific evidence, dashboard, demo, final docs/report and final release gate.
