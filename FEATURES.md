# Verified Feature Matrix

This document describes the current Phase 10 release-candidate architecture. Historical claims from earlier prototype phases are preserved in `handoff/` for audit history and are not treated as current product evidence.

## Request security path

- nginx + ModSecurity process-level edge integration
- Swavlamban gateway inspection after WAF edge processing
- HTTP request normalization and versioned feature extraction
- deterministic rule/signature checks
- supervised, unsupervised and behavioural ML signals in the decision path
- explicit ALLOW/BLOCK decisions
- protected-upstream enforcement evidence
- request-size and response-size limits
- configurable rate limiting
- upstream timeout/unavailability handling

## ML and explainability

- versioned HTTP feature schema
- supervised detector
- unsupervised anomaly detector
- behavioural anomaly detection
- structured decision evidence
- detector contributions/reasons
- model/dataset/baseline provenance
- bounded risk scoring
- synthetic evaluation corpus explicitly labelled as synthetic
- champion/challenger learning controls from the previous learning-control phases

## Managed security rules

The current lifecycle is:

```text
ML/evidence signal
  -> candidate rule
  -> replay validation
  -> authenticated human approval
  -> deployment
  -> enforcement
  -> rollback
```

Each managed rule carries lifecycle state, provenance, matcher details and deployment/audit information.

## Dashboard and operator controls

- authenticated operator dashboard
- runtime threat/event views
- model and telemetry state
- managed-rule lifecycle controls
- dynamic release/evidence state
- real-time WebSocket interface with authentication
- dashboard/demo screenshots and recording artifacts generated during Phase 10

## Learning and telemetry

- bounded asynchronous telemetry dispatcher
- privacy-safe persisted telemetry
- source identity hashing
- reviewed feedback path
- baseline/drift/challenger/promote/rollback controls
- model and feature version metadata

## Verification

The Phase 10 release gate runs from a clean GitHub Actions checkout and executes the regression suite, compilation, deterministic scenario checks, TLS termination, process-level ModSecurity enforcement, managed-rule replay, bounded load test, dashboard/demo generation and audit/handoff packaging.

The latest verified run reports 95/95 regression tests passing and a 100% result for the defined Phase 10 release checks.

## Evidence boundaries

Measured local evidence does not equal Internet-scale capacity. The repository does not claim public certificate issuance/rotation, public Internet HTTPS verification, or physical million-request distributed load. The bounded load harness reports achieved local throughput and latency. ML evaluation data used for deterministic CI is synthetic and must not be represented as field accuracy.
