# WAF Architecture Contract - Phase 1

Phase 1 defines the production-oriented architecture boundary. It does not claim that the complete WAF, ML detector, rule deployment system, or continuous-learning system is implemented.

## Fast path

proxy -> normalization/features -> rules -> ML inference -> decision

Fast-path rules: no durable database writes, no dashboard work, no model training, no unbounded network fan-out, bounded request/body sizes, deterministic stage ordering, typed inputs/outputs, and preserved evidence for explainability.

## Slow path

decision/event -> bounded queue -> telemetry/storage -> analytics -> feedback -> retraining/evaluation

The slow path never mutates an already-produced request decision.

## Dependency direction

core -> proxy/features/rules/ml/engine -> api/telemetry/storage -> dashboard
training/evaluation consume versioned evidence/data/model contracts.

waf.core uses only the Python standard library. Higher layers may depend on core; core must not depend on application/storage/dashboard implementations.

## Scale truth boundary

Millions-of-requests capacity is a future measured target, not a Phase-1 claim. The architecture keeps request-time computation bounded and durable analytics asynchronous so the fast-path service can be horizontally replicated.

## Security truth boundary

Raw request material is not automatically durable data. Later phases must define redaction, retention, authorization and secret handling before persistence.

## Acceptance

1. Major subsystem boundaries are defined.
2. Typed request/decision/evidence contracts exist.
3. Fast-path composition is deterministic and side-effect-free at this layer.
4. Slow-path publication is separate.
5. Configuration is environment-driven and bounded.
6. Architecture/dependency rules are documented.
7. Automated tests cover the contracts.