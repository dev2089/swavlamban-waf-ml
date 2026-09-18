# Next Phase: Phase 5

Phase 4 is independently verified for its defined ML milestone.

Phase 5 goal: make every security decision explainable, auditable and reproducible without unnecessary raw-payload retention.

Required capabilities:
- Stable evidence for signature, supervised, anomaly and behavioural signals.
- Decision trace tied to request ID and pipeline/model/schema versions.
- Serializable bounded evidence suitable for logs and dashboard use.
- No unnecessary raw payload persistence.
- Deterministic explanations for repeated inputs.
- Known signature, unseen anomaly and behavioural explanations.
- Edge and reverse-proxy evidence propagation.
- Performance measurement with evidence collection enabled.
- Tests for trace completeness, schema/version correlation, bounds and privacy/data minimization.
- Durable waf/database state, audit and journal.
- 100% Phase 5 gate before declaring completion.

Carry-forward Phase 4 limitations: synthetic ML evaluation data, process-local behavioural state, no semi-supervised model, no TLS milestone, no separate ModSecurity/Coraza verification, no production storage/auth, no million-request/multi-node proof.
