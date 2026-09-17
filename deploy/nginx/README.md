# Phase 10 open-source WAF integration

The reference path is:

`client -> nginx + ModSecurity -> Swavlamban ML gateway -> protected upstream`

Nginx performs open-source ModSecurity inspection first. The ML gateway then runs `EdgeWAF.analyze()` and forwards only non-blocked traffic. The Phase 10 smoke test proves that a blocked request returns before the protected upstream's success marker is reached.

The CI runner installs `libnginx-mod-http-modsecurity` and discovers the module path dynamically. The repository does not vendor the module or OWASP CRS binaries. The tracked ModSecurity policy is intentionally minimal and deterministic for the hackathon acceptance test.

The ML gateway is independently runnable with `python -m waf.gateway.proxy` and exposes `GET /__waf_health` for readiness verification.
