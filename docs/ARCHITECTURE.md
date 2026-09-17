# Target Architecture - Phase 4

## North-star architecture

```text
Internet / client
      |
      v
[Reverse proxy + open-source WAF adapter]
      |
      v
[Request normalization + bounded inspection]
      |
      +--> [Signature/rule detector]
      +--> [Supervised ML]
      +--> [Unsupervised anomaly ML]
      +--> [Learned behaviour ML]
                    |
                    v
             [Decision Policy]
                    |
             +------+------+
             |             |
           ALLOW       BLOCK/ALERT
             |             |
             +------+------+
                    v
             [Decision Event]
                    |
             +------+------+----------------+
             |             |                |
          telemetry    dashboard       feedback
             |                              |
         storage                   training/eval
                                            |
                                      model registry
```

## Active fast path

`RequestEnvelope -> http-v2 FeatureVector -> signature + supervised + unsupervised + behaviour -> DecisionResult`

The Phase 4 ML layer is connected to the same canonical request/feature contracts used by the live Phase 2/3 edge. The deterministic signature detector remains authoritative for known attack signatures.

## Phase 4 ML components

- Supervised `HistGradientBoostingClassifier`.
- Benign-only `OneClassSVM` anomaly detector with threshold learned from a benign baseline.
- Learned stateful `LogisticRegression` behavioural detector over per-source sliding-window features.
- Versioned artifact `phase4-model-v1` with schema `http-v2` and model version `phase4-ml-v1`.

## Runtime isolation

Supervised/anomaly model objects can be safely shared because they are immutable during request scoring. Behavioural request-window state is created fresh per `EdgeWAF` runtime and is not serialized into the model artifact.

## Risk semantics

Scores are risk scores, not calibrated probabilities. Known signature hits force risk to `1.0`. Non-signature risk combines detector scores using explicit Phase 4 weights: supervised `0.55`, unsupervised `0.30`, behaviour `0.15`, with the strongest signal also contributing to the final risk.

## Phase boundaries

Phase 1 established contracts.
Phase 2 established live HTTP interception and enforcement.
Phase 3 established bounded `http-v2` HTTP representation.
Phase 4 established supervised, unsupervised and learned behavioural ML on that representation.

Later phases add expanded explainability, ML-generated rule lifecycle, baseline/feedback/drift/retraining, production telemetry/storage/auth, complete scenario evidence, dashboard migration, demo and final submission.

## Critical honesty boundary
The Phase 4 ML metrics were produced from deterministic synthetic data/workloads and are reproducibility evidence only. They are not real-world Internet WAF accuracy or production capacity claims.
