# Next Phase: Phase 6

Phase 5 is independently verified for explainability and decision evidence.

Phase 6 goal: safely convert model/security evidence into candidate WAF rules with validation, approval and controlled deployment.

Required:
- Versioned candidate rule schema tied to evidence/model/feature versions.
- Syntax and semantic safety validation.
- Offline regression corpus before activation.
- Explicit approval state and immutable audit trail.
- Shadow/observe mode before enforcement.
- Atomic activation and rollback.
- Active-ruleset version metadata.
- Reproducible rule-generation and deployment evidence.
- Durable waf/database state, audit and journal.
- 100% Phase 6 gate before completion.

Carry forward: semi-supervised ML, TLS, ModSecurity/Coraza, production storage/auth, scale/failure matrix, dashboard and final submission.