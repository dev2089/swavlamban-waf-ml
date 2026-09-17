# WAF Phase 1 Architecture Foundation

The new `waf/` package is a dependency-light security core added beside the legacy prototype.

## Runtime boundary

`RequestEnvelope -> FeatureExtractor -> Detectors -> DecisionPolicy -> EventSink`

The pipeline has no web-server, database, network, or dashboard dependency. Phase 2 attaches a real reverse-proxy/open-source WAF adapter without changing the security contracts.

## Phase 1 guarantees

- one canonical request object;
- versioned feature schema;
- typed detector signals with bounded scores/confidence;
- deterministic allow/block/alert decision;
- event schema separated from persistence;
- body-size bound before feature extraction;
- no secrets in the core package;
- automated tests for the architecture seam.

## Not claimed yet

Real traffic interception, TLS termination, production WAF enforcement, supervised model training, behavioural windows, continuous retraining, distributed telemetry, authentication/RBAC and production deployment are later phases.
