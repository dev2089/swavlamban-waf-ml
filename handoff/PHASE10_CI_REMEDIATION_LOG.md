# Phase 10 CI Remediation Log

## Failure observed
- Workflow: `phase10-release-candidate`
- Run: `35255164974` (run number 52)
- Commit: `64846991ad9db10bbcf6f2d052e102037aa5a8cd`
- Job: `105316886098` (`release-gate`)
- Failing step: `Strict Phase 10 master exam`
- Failure: gateway readiness timed out on `http://127.0.0.1:18081/__waf_health` with connection refused.
- Earlier dependency installation and compile stages passed.

## Root cause established
The CI checkout intentionally does not contain `models/phase4_models.joblib` because `.gitignore` excludes `models/*.joblib`. `EdgeWAF` therefore entered its fallback path and trained the full default ML ensemble while the gateway process was starting. The master exam only waited 15 seconds for readiness and did not expose the child gateway log when that wait expired. The result was a misleading readiness failure instead of a useful startup diagnostic.

This is a CI/runtime readiness bug, not a missing connector problem. Supabase is already live and verified. Vercel is not part of this execution path.

## Remediation applied
1. The Phase 10 workflow now materializes the deterministic model artifact in `/tmp` and copies only the ignored `.joblib` into `models/phase4_models.joblib` before the master exam. The tracked manifest is not modified.
2. The master exam readiness wait is expanded to 90 seconds.
3. The wait path now checks the child process state and includes the latest upstream/gateway log tails in a startup failure.
4. The first failure is preserved as evidence and is not relabeled as a successful check.

## Verification rule
Phase 10 is not considered CI-complete until a fresh Phase 10 Actions run reaches the master exam, passes every acceptance check, builds the binary artifacts, builds the portable handoff, and uploads the final auditor bundle.
