# Phase 10 CI Remediation Plan

The release gate failure was narrowed to gateway startup readiness in GitHub Actions. The runner successfully installed Python dependencies, nginx/ModSecurity/CRS, ffmpeg, Chromium and compiled the project. The failure occurred when the master exam waited for `127.0.0.1:18081/__waf_health`.

The gateway's `EdgeWAF` loads `models/phase4_models.joblib`. The repository deliberately ignores this generated artifact, so a clean CI checkout falls back to training the default ensemble during gateway process startup. The original 15-second readiness wait did not account for this CI startup path and did not surface the child-process logs.

Remediation:

1. Prebuild the deterministic Phase 4 model artifact in CI into `/tmp` and copy only the ignored `.joblib` into `models/phase4_models.joblib`.
2. Extend gateway readiness to 90 seconds.
3. Fail early when the gateway child exits and include its log tail in the error.
4. Retain this failure and remediation history as permanent Phase 10 evidence.
5. Require a fresh green GitHub Actions release-gate before Phase 10 is marked complete.
