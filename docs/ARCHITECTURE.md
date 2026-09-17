# Target Architecture - Phase 3

## North-star architecture

```text
Internet / client
      |
      v
[Reverse proxy + open-source WAF adapter]
      |
      v
[Request normalization + bounded inspection]   <-- Phase 3
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

## Active fast path

`RequestEnvelope -> http-v2 FeatureVector -> detector signals -> DecisionResult`

The Phase 3 feature extractor is deterministic, versioned and bounded. It performs NFKC normalization, path/query-safe multi-pass URL decoding, bounded query parsing, normalized header inspection, body decoding and numeric feature generation.

## http-v2 properties

- 38 numeric features.
- Every feature is clamped to `[0,1]`.
- No raw payload is stored in the feature vector.
- URL decoding is capped at three passes.
- Query parsing is capped at 256 fields.
- Header values are capped at 4 KiB for feature extraction.
- Body scanning is capped at 256 KiB.
- Path and query use separate decoding semantics to preserve path `+` characters.

## Phase boundaries

Phase 1 established security-core contracts.
Phase 2 established live HTTP interception and real pre-forwarding enforcement.
Phase 3 establishes the production HTTP feature layer used by the edge WAF.

Later phases add production ML, behaviour windows, continuous learning, production storage/authentication, dashboard migration and final challenge evidence.

## Design rules

1. Transport adapters may change; security contracts remain stable.
2. Feature/model schemas are explicitly versioned.
3. Risk scores are normalized characteristics, not probabilities.
4. Decisions are deterministic for fixed inputs and detector outputs.
5. Size and parsing limits are enforced before expensive inspection.
6. Persistence is asynchronous and outside the decision-critical path.
7. The core feature vector contains no secrets or raw request content.
