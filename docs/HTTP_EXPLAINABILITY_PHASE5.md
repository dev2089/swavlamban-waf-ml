# Phase 5 Explainability Architecture

RequestEnvelope -> http-v2 -> signatures + Phase 4 ML -> EdgeDecisionPolicy -> DecisionEvidence -> event-v2

The existing security decision is computed first. Evidence is attached after policy evaluation so Phase 4 enforcement semantics remain unchanged.

Evidence contains:
- request ID
- final decision and risk score
- detector scores, confidence, reasons and rule IDs
- detector contribution estimates
- six feature groups covering the 40-feature http-v2 schema
- numeric feature snapshot
- supervised/anomaly group-level perturbation attribution
- behavioural risk evidence
- model/feature/dataset/baseline/pipeline/ruleset provenance
- deterministic human-readable explanation
- privacy declarations

Privacy:
Raw request payload, raw query string, raw header values, source IP and host text are not retained in the evidence object. The only request correlation identifier is request_id.

Attribution:
For supervised and anomaly models, each feature group is zeroed in a copy of the numeric vector and the resulting risk change is recorded. This is deterministic lightweight evidence and is not presented as causal feature importance.

Failure:
A model inference exception is already converted by the Phase 4 edge to a risk=1.0 fail-closed signal. Phase 5 explains that path without requiring trained model components.

Limitations:
Evidence generation adds runtime cost. Behavioural state remains process-local. Production persistence/authentication, richer access controls and later telemetry storage belong to subsequent phases.
