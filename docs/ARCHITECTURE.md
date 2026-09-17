# Target Architecture and Phase 1 Implementation

## North-star architecture

```text
Internet / client
      |
      v
[Reverse proxy + open-source WAF adapter]     <-- Phase 2
      |
      v
[Request normalization]
      |
      +--> [Signature/rule detectors]
      +--> [ML detectors]
      +--> [Behaviour detector]
              |
              v
        [Decision Policy]
              |
        +-----+-----+
        |           |
      ALLOW       BLOCK/ALERT
        |           |
        +-----+-----+
              v
       [Async telemetry]
          /        \\
      storage    dashboard
              |
         [feedback]
              |
        [training/eval]
              |
        [model registry]
```

## Phase 1 boundaries

The new `waf/` package implements only the security-core seam. It does not intercept network traffic and it does not claim to be the finished WAF.

### Fast path

`RequestEnvelope -> FeatureVector -> DetectionSignal[] -> DecisionResult`

No database writes, network calls, or dashboard work are required to make a decision.

### Slow path

`DecisionResult -> event schema -> sink -> persistent telemetry` will be attached in later phases.

## Design rules

1. Transport adapters may change; security contracts must not.
2. Every ML feature/model has an explicit version.
3. Scores are normalized to `[0,1]`; they are risk scores, not probabilities.
4. A decision is deterministic for fixed request and detector outputs.
5. Oversized bodies are bounded before feature extraction.
6. Persistence is an adapter, not a prerequisite of a security decision.
7. The core package contains no credentials and no browser-facing security policy.

## Migration strategy

The legacy `backend/server.py`, `ml_model.py`, and dashboard remain untouched in Phase 1. Phase 2 will migrate behaviour behind these contracts and then retire duplicate paths after parity tests pass.
