# Challenge 3 Presentation Source

## Slide 1 — Swavlamban WAF ML
ML-integrated open-source WAF with explainable decisions, controlled learning and secure operations.

## Slide 2 — The security problem
Web requests are noisy, adversarial and constantly changing. A useful WAF needs deterministic controls plus adaptive ML signals without turning learning into uncontrolled deployment.

## Slide 3 — Detection pipeline
Request normalization → deterministic detectors → ML risk score → policy decision → structured evidence → sanitized telemetry.

## Slide 4 — Explainable decisions
Every decision can carry detector contributions, feature groups, reasons, rule identifiers and model/rule version provenance under `evidence-v1`.

## Slide 5 — Managed rule lifecycle
Generate → validate → human approval → deploy → enforce → rollback. No-auto-promotion remains the safety boundary.

## Slide 6 — Learning control
Baseline → reviewed feedback → drift analysis → challenger training → champion/challenger evaluation → explicit human promotion or rollback.

## Slide 7 — Production security
Signed bearer auth, explicit RBAC, fail-closed production configuration, privacy-safe audit, secret hygiene, RLS and server-only Supabase access.

## Slide 8 — Live data and edge evidence
Supabase schema/RLS/privileges are live-verified. Local nginx TLS integration demonstrates HTTPS termination and SQL blocking. Raw request material is not retained in runtime telemetry.

## Slide 9 — Challenge scenarios
Benign traffic is allowed. Known SQL/XSS and command variants are blocked. API burst behaviour is exercised. A deterministic 500-request in-process benchmark measures runtime overhead within a stated boundary.

## Slide 10 — Evidence boundary and release
The release candidate is reproducible and fully logged. Public certificate operations, ModSecurity/Coraza installation and Internet-scale distributed load remain explicitly unverified rather than being presented as facts.
